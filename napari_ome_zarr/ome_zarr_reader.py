# zarr v3

from abc import ABC
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import zarr
from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
from napari.utils.transforms import Affine
from zarr import Group
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import SyncMixin

import concurrent.futures as futures

import time

from .plate import get_first_field_path, get_first_well, get_pyramid_lazy

import math
from zarr.codecs.bytes import BytesCodec

# Add a small patch to Zarr's BytesCodec to handle 4096-byte alignment padding in OME-Zarr chunk buffers.
# Dynamically patch based on the Zarr version installed
if hasattr(BytesCodec, "_decode_sync"):
    # Zarr v3.1.6+
    _orig_decode_sync = BytesCodec._decode_sync

    def _padded_decode_sync(self, chunk_bytes, chunk_spec):
        expected_bytes = math.prod(chunk_spec.shape) * chunk_spec.dtype.item_size
        if len(chunk_bytes) > expected_bytes:
            chunk_bytes = chunk_bytes[:expected_bytes]
        return _orig_decode_sync(self, chunk_bytes, chunk_spec)

    BytesCodec._decode_sync = _padded_decode_sync

elif hasattr(BytesCodec, "_decode_single"):
    # Zarr v3.1.5
    _orig_decode_single = BytesCodec._decode_single

    async def _padded_decode_single(self, chunk_bytes, chunk_spec):
        expected_bytes = math.prod(chunk_spec.shape) * chunk_spec.dtype.item_size
        if len(chunk_bytes) > expected_bytes:
            chunk_bytes = chunk_bytes[:expected_bytes]
        return await _orig_decode_single(self, chunk_bytes, chunk_spec)

    BytesCodec._decode_single = _padded_decode_single
# StrDict = Dict[str, Any]
# LayerData = Union[Tuple[Any], Tuple[Any, StrDict], Tuple[Any, StrDict, str]]
LayerData = Tuple[List[da.core.Array], Dict[str, Any], str]


def _match_colors_to_available_colormap(custom_cmap: Colormap) -> Colormap:
    """Helper function to match Colormap to an existing napari Colormap.
    If the colormap matches, return the specific napari Colormap, otherwise return the
    the original Colormap.
    """
    for available_cmap in AVAILABLE_COLORMAPS.values():
        if (
            np.array_equal(available_cmap.controls, custom_cmap.controls)
            and np.array_equal(available_cmap.colors, custom_cmap.colors)
            and available_cmap.interpolation == custom_cmap.interpolation
        ):
            custom_cmap = available_cmap
            break

    return custom_cmap


def remove_axis_from_transform(transform: Dict[str, Any], axis: int) -> Dict[str, Any]:
    """Remove a specific axis from an OME-Zarr transform dict."""
    new_transform = transform.copy()
    if transform["type"] == "scale":
        new_scale = transform["scale"][:]
        del new_scale[axis]
        new_transform["scale"] = new_scale
    if transform["type"] == "translation":
        new_translation = transform["translation"][:]
        del new_translation[axis]
        new_transform["translation"] = new_translation
    if transform["type"] == "rotation":
        matrix = np.array(transform["rotation"])
        matrix = np.delete(matrix, axis, 0)  # remove row
        matrix = np.delete(matrix, axis, 1)  # remove column
        new_transform["rotation"] = matrix.tolist()
    if transform["type"] == "affine":
        matrix = np.array(transform["affine"])
        matrix = np.delete(matrix, axis, 0)  # remove row
        matrix = np.delete(matrix, axis, 1)  # remove column
        new_transform["affine"] = matrix.tolist()
    if transform["type"] == "sequence":
        new_transforms = []
        for sub_transform in transform["transformations"]:
            new_sub_transform = remove_axis_from_transform(sub_transform, axis)
            new_transforms.append(new_sub_transform)
        new_transform["transformations"] = new_transforms
    return new_transform


def transform_to_affine(transform: Dict[str, Any]) -> Affine:
    """Convert a single OME-Zarr transform dict to an Affine object."""
    aff: Affine = None
    if transform["type"] == "scale":
        aff = Affine(scale=transform["scale"])
    if transform["type"] == "translation":
        aff = Affine(translate=transform["translation"])
    if transform["type"] == "rotation":
        matrix = np.array(transform["rotation"])
        # Spec says that rotation matrix is 1 row and column smaller than affine
        matrix = np.pad(
            matrix,
            pad_width=((0, 1), (0, 1)),
            mode="constant",
            constant_values=0,
        )
        matrix[-1, -1] = 1
        aff = Affine(affine_matrix=matrix)
    if transform["type"] == "affine":
        matrix = np.array(transform["affine"])
        # Spec says that rotation matrix is 1 row smaller than affine
        matrix = np.pad(
            matrix,
            pad_width=((0, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        matrix[-1, -1] = 1
        aff = Affine(affine_matrix=matrix)
    return aff


def transforms_to_affine(
    transforms: List[Dict[str, Any]], channel_axis: int | None
) -> Affine:
    # first unwrap and flatten any 'sequence' transforms...
    # NB: if any 'sequence' contains another 'sequence' this is ignored.
    flat_transforms: List[Dict[str, Any]] = []
    for transf in transforms:
        if transf["type"] == "sequence":
            flat_transforms.extend(transf["transformations"])
        else:
            flat_transforms.append(transf)

    # Don't create Affine until we know dimensions...
    aff: Affine = None
    for transf in flat_transforms:
        # print("transforms_to_affine..........ch,transf", channel_axis, transf)
        trans_aff = transform_to_affine(transf)
        if aff is None:
            aff = trans_aff
        elif trans_aff is not None:
            aff = trans_aff.compose(aff)
    # finally, remove channel axis from 2D matrix
    if channel_axis is not None:
        matrix = aff.affine_matrix
        for dim in (0, 1):
            matrix = np.delete(matrix, channel_axis, dim)
        aff = Affine(affine_matrix=matrix)
    return aff


class Spec(ABC):
    def __init__(self, group: Group) -> None:
        self.group = group
        self.parent_transforms: List[Dict[str, Any]] = []

    @staticmethod
    def matches(group: Group) -> bool:
        return False

    def data(self) -> List[da.core.Array]:
        return []

    def metadata(self) -> Dict[str, Any]:
        # napari layer metadata
        return {}

    def children(self) -> list["Spec"]:
        return []

    def iter_nodes(self) -> Iterable["Spec"]:
        yield self
        for child in self.children():
            yield from child.iter_nodes()

    def iter_data(self) -> Iterable[da.core.Array]:
        for node in self.iter_nodes():
            data = node.data()
            if data:
                yield data

    @staticmethod
    def get_attrs(group: Group) -> dict:
        if "ome" in group.attrs:
            return group.attrs["ome"]
        return group.attrs


class Multiscales(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        return "multiscales" in Spec.get_attrs(group)

    def children(self) -> list[Spec]:
        ch: list[Spec] = []
        # test for child "labels"
        try:
            grp = self.group["labels"]
            attrs = Spec.get_attrs(grp)
            if "labels" in attrs:
                for name in attrs["labels"]:
                    g = grp[name]
                    if Label.matches(g):
                        label_image = Label(g)
                        # Label inherits parent transforms
                        ch_axis = self.metadata().get("channel_axis", None)
                        for transf in self.parent_transforms:
                            label_image.add_parent_transform(transf, ch_axis)
                        ch.append(label_image)
        except KeyError:
            pass
        return ch

    def data(self) -> list[da.core.Array]:
        attrs = Spec.get_attrs(self.group)
        paths = [ds["path"] for ds in attrs["multiscales"][0]["datasets"]]
        return [da.from_zarr(self.group[path]) for path in paths]

    def metadata(self) -> Dict[str, Any]:
        rsp: dict = {}
        attrs = Spec.get_attrs(self.group)
        # TODO: handle v0.6 AND v0.5 and earlier...
        if "axes" in attrs["multiscales"][0]:
            axes = attrs["multiscales"][0]["axes"]
        else:
            axes = attrs["multiscales"][0]["coordinateSystems"][0]["axes"]
        dataset_0 = attrs["multiscales"][0]["datasets"][0]
        channel_axis = None
        atypes = [axis["type"] for axis in axes]
        if "channel" in atypes:
            channel_axis = atypes.index("channel")
            rsp["channel_axis"] = channel_axis

        transforms = []

        # if we have "graph" of transforms from scene...
        if len(self.parent_transforms) > 0:
            transforms.extend(self.parent_transforms)
        else:
            # First we handle transforms from datasets[0]...
            if "coordinateTransformations" in dataset_0:
                transforms.extend(dataset_0["coordinateTransformations"])
            # Then check for coordinateTransformations at top level
            if "coordinateTransformations" in attrs["multiscales"][0]:
                transforms.extend(attrs["multiscales"][0]["coordinateTransformations"])

        # compile all transforms into single Affine
        print("\nALL transforms:", transforms)
        rsp["affine"] = transforms_to_affine(transforms, channel_axis)

        if "omero" in attrs:
            colormaps = []
            ch_names = []
            visibles = []
            contrast_limits: list[list[int]] = []
            model = attrs["omero"].get("rdefs", {}).get("model", "unset")
            greyscale = model == "greyscale"

            for index, ch in enumerate(attrs["omero"]["channels"]):
                color = ch.get("color", None)
                if color is not None:
                    rgb = [(int(color[i : i + 2], 16) / 255) for i in range(0, 6, 2)]
                    if greyscale:
                        rgb = [1, 1, 1]
                    # colormap is range: black -> rgb color
                    cm = Colormap([[0, 0, 0], rgb])
                    # Try to match colormap to an existing napari colormap
                    cm = _match_colors_to_available_colormap(cm)
                    colormaps.append(cm)
                ch_names.append(ch.get("label", f"channel_{index}"))
                visibles.append(ch.get("active", True))

                window = ch.get("window", None)
                if window is not None:
                    start = window.get("start", None)
                    end = window.get("end", None)
                    if start is not None and end is not None:
                        # skip if None. Otherwise check no previous skip
                        if len(contrast_limits) == index:
                            contrast_limits.append([start, end])

            if rsp.get("channel_axis") is not None:
                rsp["colormap"] = colormaps
                rsp["name"] = ch_names
                if len(contrast_limits) > 0:
                    rsp["contrast_limits"] = contrast_limits
                rsp["visible"] = visibles
            else:
                rsp["colormap"] = colormaps[0]
                rsp["name"] = ch_names[0]
                if len(contrast_limits) > 0:
                    rsp["contrast_limits"] = contrast_limits[0]
                rsp["visible"] = visibles[0]

        return rsp


class Bioformats2raw(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        # Don't consider "plate" as a Bioformats2raw layout
        return "bioformats2raw.layout" in attrs and "plate" not in attrs

    def children(self) -> list[Spec]:
        # lookup children from series of OME/METADATA.xml
        xml_data = SyncMixin()._sync(
            self.group.store.get(
                "OME/METADATA.ome.xml", prototype=default_buffer_prototype()
            )
        )
        # print("xml_data", xml_data.to_bytes())
        root = ET.fromstring(xml_data.to_bytes())
        rv: list[Spec] = []
        for child in root:
            # {http://www.openmicroscopy.org/Schemas/OME/2016-06}Image
            print(child.tag)
            node_id = child.attrib.get("ID", "")
            if child.tag.endswith("Image") and node_id.startswith("Image:"):
                print("Image ID", node_id)
                image_path = node_id.replace("Image:", "")
                g = self.group[image_path]
                if Multiscales.matches(g):
                    rv.append(Multiscales(g))
        return rv

    # override to NOT yield self since node has no data
    def iter_nodes(self) -> Iterable[Spec]:
        for child in self.children():
            yield from child.iter_nodes()


def path_name(in_out: dict | str) -> str:
    # helper to get path from 'input' or 'output' dict
    if isinstance(in_out, str):
        return in_out
    paths = []
    if "path" in in_out:
        paths.append(in_out["path"])
    if "name" in in_out:
        paths.append(in_out["name"])
    return "/".join(paths)


class Scene(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        attrs = Spec.get_attrs(group)
        print("Scene.matches", attrs)
        return "scene" in attrs

    def children(self) -> list[Spec]:
        # lookup children from coordinateTransformations 'input' paths
        rv: list[Spec] = []
        scene_attrs = Spec.get_attrs(self.group).get("scene", {})
        transfs = scene_attrs.get("coordinateTransformations", [])
        output = None
        for transf in transfs:
            # TODO: input may NOT have "path"
            # TODO: input "name" is ignored!
            image_path = transf.get("input").get("path", None)
            out = transf.get("output")
            if output is None:
                output = out
            elif path_name(output) != path_name(out):
                print(
                    f"WARNING: '{path_name(out)}' output different"
                    f"from previous '{path_name(output)}'"
                )
                continue
            g = self.group[image_path]
            print("scene child", image_path, g)
            if Multiscales.matches(g):
                ms_image = Multiscales(g)
                # child image gets the parent transform
                # TODO: use the input 'name' to specify which coordinateSystem to use
                ms_image.parent_transforms.append(transf)
                rv.append(ms_image)

        # check if 'output' is path to image...
        try:
            if output is not None:
                g = self.group[output.get("path", "")]
                if Multiscales.matches(g):
                    # Add to start (layer behind other tiles)
                    rv.insert(0, Multiscales(g))
        except KeyError:
            pass
        return rv

    # override to NOT yield self since node has no data
    def iter_nodes(self) -> Iterable[Spec]:

        # transforms key is each transform output "path.zarr/name"
        # (where name is name of coordinateSystem)
        transforms = defaultdict(list)  # type: Dict[str, List[Dict[str, Any]]]
        # track unique coordinateSystems by "path.zarr/name"
        # systems = set()

        # FIRST, go through all transforms in this scene,
        # AND any child transforms we find at 'input' or 'output' paths...
        scene_attrs = Spec.get_attrs(self.group).get("scene", {})
        image_attrs = {}
        for transf in scene_attrs.get("coordinateTransformations", []):
            output = transf["output"]
            transf["input_full_path"] = path_name(transf["input"])
            transforms[path_name(output)].append(transf)
            # traverse to input/output coordinateSystem paths...
            for io in ("input", "output"):
                image_path = transf[io].get("path", None)
                if image_path is not None and image_path not in image_attrs:
                    image_attrs[image_path] = Spec.get_attrs(self.group[image_path])
                    # need to add child transforms to our graph
                    for ms in image_attrs[image_path].get("multiscales", []):
                        for child_transf in ms.get("coordinateTransformations", []):
                            # TODO: assert output doesn't have 'path'?
                            out_name = path_name(child_transf["output"])
                            out_path_name = image_path + "/" + out_name
                            input_path_name = (
                                image_path + "/" + path_name(child_transf["input"])
                            )
                            child_transf["input_full_path"] = input_path_name
                            transforms[out_path_name].append(child_transf)
                        # and the datasets... - find 'output' (just use first one)
                        for ds in ms.get("datasets", [])[:1]:
                            # only expect single transform...
                            for ds_transf in ds.get("coordinateTransformations", []):
                                out_name = path_name(ds_transf["output"])
                                out_path_name = image_path + "/" + out_name
                                # don't expect this to be used. e.g. image.zarr/s0
                                input_path_name = (
                                    image_path + "/" + path_name(ds_transf["input"])
                                )
                                ds_transf["input_full_path"] = input_path_name
                                transforms[out_path_name].append(ds_transf)
                                # Use this to create the Multiscales object below...
                                ds_transf["multiscale_path"] = image_path

        # print("Scene.iter_nodes...transforms")
        # for key, transfs in transforms.items():
        #     print(f"  {key}: ", [t["input_full_path"] for t in transfs])
        #   translated_x_and_y:  ['4995115_full.zarr/physical', 'translated_x50']
        #   4995115_full.zarr/physical:  ['4995115_full.zarr/s0']
        #   translated_x50:  ['rot10.zarr/rotated', 'rot45.zarr/rotated']
        #   rot10.zarr/rotated:  ['rot10.zarr/physical']
        #   rot10.zarr/physical:  ['rot10.zarr/s0']
        #   rot45.zarr/rotated:  ['rot45.zarr/physical']
        #   rot45.zarr/physical:  ['rot45.zarr/s0']

        # find the unique coordinateSystems (outputs)
        outputs = set(transforms.keys())

        # remove any output that is also an input (i.e. intermediate node in graph)
        for transf_list in transforms.values():
            outputs -= {t["input_full_path"] for t in transf_list}

        print("Scene.iter_nodes...final outputs ----->", outputs)

        assert len(outputs) == 1, "Multiple output coordinateSystems not supported"

        # now we can iterate through the graph starting at the unique output
        # coordinateSystem, and yield any nodes we find along the way...
        def iter_graph(
            path_name: str, parent_trans: list[Dict[str, Any]]
        ) -> Iterable[list[Dict[str, Any]]]:
            # find transforms that output to this node
            node_transfs = transforms.get(path_name)
            # print("iter_graph", path_name, parent_trans)

            if node_transfs is None:
                yield parent_trans
            # find the group for this node
            else:
                for transf in node_transfs:
                    parents_copy = parent_trans[:]
                    parents_copy.append(transf)
                    yield from iter_graph(transf["input_full_path"], parents_copy)

        output = outputs.pop()
        inputs = list(iter_graph(output, []))

        print("IMAGES")

        # Ignore any transform lists that don't lead to a multiscale image...
        multiscale_inputs = [inp for inp in inputs if "multiscale_path" in inp[-1]]

        for trans_list in multiscale_inputs:
            print("---------------------")
            print(trans_list[-1]["multiscale_path"])
            # the last transform should have "multiscale_path" key...
            ms_image = Multiscales(self.group[trans_list[-1]["multiscale_path"]])

            # reverse, so we START at datasets -> coordinateSystem -> scene
            trans_list.reverse()
            ms_image.parent_transforms = trans_list
            yield ms_image

        # for child in self.children():
        #     yield from child.iter_nodes()


class FusedScene(Spec):
    """
    A replacement for Scene that fuses all multiscale images into a single 
    Dask virtual canvas per zoom level, returning only ONE layer to Napari.
    """
    @staticmethod
    def matches(group: zarr.Group) -> bool:
        attrs = Spec.get_attrs(group)
        return "scene" in attrs

    def __init__(self, group: zarr.Group):
        super().__init__(group)
        self.fused_pyramid: List[da.core.Array] = []
        self.fused_metadata: Dict[str, Any] = {}
        self._build_fused_scene()

    def _build_fused_scene(self):
        """Pre-computes the bounding box and constructs the lazy Dask canvas."""
        scene_attrs = Spec.get_attrs(self.group).get("scene", {})
        transfs = scene_attrs.get("coordinateTransformations", [])

        images_info = []
        base_metadata = None

        parse_start = time.perf_counter()

        def parse_tile(transf):
            image_path = transf.get("input", {}).get("path", None)
            if not image_path:
                return None
            
            try:
                # Heavy I/O is safely handled by the threads
                g = self.group[image_path]
                if Multiscales.matches(g):
                    ms_image = Multiscales(g)
                    pyramid = ms_image.data()
                    pyramid.sort(key=lambda arr: np.prod(arr.shape), reverse=True)
                    
                    return {
                        "path": image_path,
                        "pyramid": pyramid,
                        "raw_transf": transf, # --- FIX: Pass the raw dictionary out ---
                        "ms_image": ms_image
                    }
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
            return None

        # Execute the I/O concurrently
        # with futures.ThreadPoolExecutor(max_workers=1000) as executor:
        #     results = list(executor.map(parse_tile, transfs))
        results = []
        for transf in transfs:
            res = parse_tile(transf)
            if res is not None:
                results.append(res)
            

        # Process the results safely in the main thread
        for res in results:
            if res is not None:
                ms_image = res["ms_image"]
                if base_metadata is None:
                    base_metadata = ms_image.metadata()

                l0_scale = [1.0, 1.0, 1.0] # Fallback default
                try:
                    # Get the raw attributes of the multiscale group
                    ms_attrs = Spec.get_attrs(ms_image.group)
                    datasets = ms_attrs.get("multiscales", [{}])[0].get("datasets", [])
                    
                    # Look at datasets[0] (Level 0) for coordinateTransformations
                    for ct in datasets[0].get("coordinateTransformations", []):
                        if ct.get("type") == "scale":
                            l0_scale = ct["scale"]
                            break
                except Exception as e:
                    print(f"Warning: Could not parse scale for {res['path']}: {e}")

                res["l0_scale"] = l0_scale # Store it for later!

                aff = transform_to_affine(res["raw_transf"])
                res["transform"] = aff.affine_matrix if aff else np.eye(3)
                
                # Clean up memory
                res.pop("ms_image", None)
                res.pop("raw_transf", None)
                images_info.append(res)

        if not images_info:
            return

        self.fused_metadata = base_metadata.copy()
        self.fused_metadata["name"] = f"Fused Scene ({len(images_info)} tiles)"
        
        self.fused_metadata.pop("affine", None)
        self.fused_metadata.pop("scale", None)

        parse_end = time.perf_counter()
        print(f"Parsed {len(images_info)} tiles in {(parse_end - parse_start) * 1000:.3f} ms")
        info_start = time.perf_counter()
        num_levels = min(len(info["pyramid"]) for info in images_info)

        l0_min_y, l0_min_x = float('inf'), float('inf')
        l0_max_y, l0_max_x = float('-inf'), float('-inf')

        for info in images_info:
            h, w = info["pyramid"][0].shape[-2:]
            y_offset = info["transform"][-3, -1]
            x_offset = info["transform"][-2, -1]

            scale_y = info["l0_scale"][-2]
            scale_x = info["l0_scale"][-1]

            y_offset = y_offset / scale_y
            x_offset = x_offset / scale_x
            
            l0_min_y = min(l0_min_y, y_offset)
            l0_min_x = min(l0_min_x, x_offset)
            l0_max_y = max(l0_max_y, y_offset + h)
            l0_max_x = max(l0_max_x, x_offset + w)

        max_scale_factor = 2 ** (num_levels - 1)
        aligned_min_y = int(np.floor(l0_min_y / max_scale_factor) * max_scale_factor)
        aligned_min_x = int(np.floor(l0_min_x / max_scale_factor) * max_scale_factor)
        aligned_max_y = int(np.ceil(l0_max_y / max_scale_factor) * max_scale_factor)
        aligned_max_x = int(np.ceil(l0_max_x / max_scale_factor) * max_scale_factor)

        l0_spatial_shape = (aligned_max_y - aligned_min_y, aligned_max_x - aligned_min_x)
        self.fused_metadata["translate"] = (aligned_min_y, aligned_min_x)

        for level in range(num_levels):
            template_tile = images_info[0]["pyramid"][level]
            prefix_shape = template_tile.shape[:-2]
            
            # --- FIX 3: Dynamic Native Ratios instead of hardcoded 2**level ---
            # This perfectly syncs the canvas shape with the actual Zarr tile reduction
            ratio_y = images_info[0]["pyramid"][0].shape[-2] / template_tile.shape[-2]
            ratio_x = images_info[0]["pyramid"][0].shape[-1] / template_tile.shape[-1]
            
            spatial_shape = (
                int(np.round(l0_spatial_shape[0] / ratio_y)),
                int(np.round(l0_spatial_shape[1] / ratio_x))
            )
            canvas_shape = prefix_shape + spatial_shape
            level_tiles = []
            for info in images_info:
                # y_offset = int(np.round(info["transform"][-3, -1] / ratio_y))
                # x_offset = int(np.round(info["transform"][-2, -1] / ratio_x))
                trans_y = info["transform"][-3, -1]
                trans_x = info["transform"][-2, -1]
                
                scale_y = info["l0_scale"][-2]
                scale_x = info["l0_scale"][-1]
                
                l0_y_offset = trans_y / scale_y
                l0_x_offset = trans_x / scale_x
                
                y_offset = int(np.round(l0_y_offset / ratio_y))
                x_offset = int(np.round(l0_x_offset / ratio_x))
                tile = info["pyramid"][level]
                t_h, t_w = tile.shape[-2:]
                
                # Store only the fast primitive integers we need for the intersection math
                level_tiles.append((y_offset, x_offset, t_h, t_w, tile))

            # --- NEW: Define a much larger chunk size for the Napari renderer ---
            # 1024 or 2048 are usually the sweet spots for GPU texture uploads
            target_spatial_chunk = (1024, 1024) 
            
            prefix_chunks = prefix_shape
            custom_chunks = prefix_chunks + target_spatial_chunk
            
            # Create the canvas using the custom larger chunks
            canvas_template = da.zeros(canvas_shape, chunks=custom_chunks, dtype=template_tile.dtype)
            #canvas_template = da.zeros(canvas_shape, chunks=template_tile.chunksize, dtype=template_tile.dtype)
            

            def render_chunk(
                block, 
                block_info=None, 
                _level_tiles=level_tiles, 
                _ratio_y=ratio_y, 
                _ratio_x=ratio_x, 
                _aligned_min_y=aligned_min_y, 
                _aligned_min_x=aligned_min_x
            ):
                if block_info is None or len(block_info) == 0:
                    return block

                # timer start
                start = time.perf_counter()
                
                loc = block_info[None]['array-location']
                y_start = loc[-2][0]
                x_start = loc[-1][0]

                
                scale_factor = 2 ** level
                global_y_start = y_start + int(np.round(_aligned_min_y // scale_factor))
                global_x_start = x_start + int(np.round(_aligned_min_x // scale_factor))
                
                chunk_data = np.zeros(block.shape, dtype=block.dtype)
                block_h, block_w = block.shape[-2:]

                fetch_tasks = []
                write_coords = []

                find_start = time.perf_counter()
                # 1. Iterate through tiles and collect the delayed Dask slices
                for y_offset, x_offset, t_h, t_w, tile in _level_tiles:
                    inter_y_start = max(global_y_start, y_offset)
                    inter_y_end = min(global_y_start + block_h, y_offset + t_h)
                    
                    # --- OPTIMIZATION 2: Fast-fail the Y-axis before calculating X ---
                    if inter_y_start >= inter_y_end:
                        continue 
                        
                    inter_x_start = max(global_x_start, x_offset)
                    inter_x_end = min(global_x_start + block_w, x_offset + t_w)
                    
                    # --- OPTIMIZATION 3: Fast-fail the X-axis ---
                    if inter_x_start >= inter_x_end:
                        continue
                        
                    out_y_start = inter_y_start - global_y_start
                    out_y_end = inter_y_end - global_y_start
                    out_x_start = inter_x_start - global_x_start
                    out_x_end = inter_x_end - global_x_start
                    
                    in_y_start = inter_y_start - y_offset
                    in_y_end = inter_y_end - y_offset
                    in_x_start = inter_x_start - x_offset
                    in_x_end = inter_x_end - x_offset
                    
                    # SLICE, BUT DO NOT COMPUTE YET
                    # This creates a delayed Dask object
                    delayed_slice = tile[..., in_y_start:in_y_end, in_x_start:in_x_end]

                    # Store the delayed task and its corresponding target coordinates
                    fetch_tasks.append(delayed_slice)
                    write_coords.append((out_y_start, out_y_end, out_x_start, out_x_end))


                find_end = time.perf_counter()
                print(f"Found {len(fetch_tasks)} overlapping tiles in {(find_end - find_start) * 1000:.3f} ms")
                
                # 2. Fetch all overlapping data in parallel
                if fetch_tasks:
                    fetch_start = time.perf_counter()
                    
                    # dask.compute resolves multiple delayed objects concurrently using its thread pool
                    fetched_data = da.compute(*fetch_tasks) 
                    
                    fetch_end = time.perf_counter()
                    print(f"Fetched {len(fetch_tasks)} tiles in parallel in {(fetch_end - fetch_start) * 1000:.3f} ms")

                    # 3. Apply the fetched data to the chunk synchronously to avoid race conditions
                    slice_start = time.perf_counter()

                    
                    for data_slice, (oy_s, oy_e, ox_s, ox_e) in zip(fetched_data, write_coords):
                        chunk_data[..., oy_s:oy_e, ox_s:ox_e] = np.maximum(
                            chunk_data[..., oy_s:oy_e, ox_s:ox_e],
                            data_slice
                        )

                    slice_end = time.perf_counter()
                    print(f"Applied {len(fetch_tasks)} tiles to chunk in {(slice_end - slice_start) * 1000:.3f} ms")

                # time.sleep(1) #check if renders are happening in parralel
                #print(f"Fetched {fetched_tiles} tiles for chunk at {loc}")
                # timer end
                end = time.perf_counter()
                print(f"Rendered chunk at {loc} in {end - start:.6f} seconds")

                return chunk_data

            fused_level = canvas_template.map_blocks(render_chunk, dtype=canvas_template.dtype)
            self.fused_pyramid.append(fused_level)
        info_end = time.perf_counter()
        print(f"Constructed fused canvas pyramid in {(info_end - info_start) * 1000:.3f} ms")

    def data(self) -> list[da.core.Array]:
        return self.fused_pyramid

    def metadata(self) -> Dict[str, Any]:
        return self.fused_metadata

    # Override iter_nodes to ONLY yield self, completely bypassing the recursive layer generation
    def iter_nodes(self) -> Iterable["Spec"]:
        yield self

class Plate(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        return "plate" in Spec.get_attrs(group)

    def data(self) -> list[da.core.Array]:
        # we want to return a dask pyramid...
        return get_pyramid_lazy(self.group)

    def metadata(self) -> dict:
        well_group = get_first_well(self.group)
        first_field_path = get_first_field_path(well_group)
        image_group = well_group[first_field_path]
        return Multiscales(image_group).metadata()

    def children(self) -> list[Spec]:
        # Plate has children If it has labels - check one Well...
        # Child is PlateLabels
        well_group = get_first_well(self.group)
        first_field_path = get_first_field_path(well_group)
        image_group = well_group[first_field_path]
        labels_group = image_group.get("labels", None)
        if labels_group is not None:
            labels_attrs = Spec.get_attrs(labels_group)
            if "labels" in labels_attrs:
                ch: list[Spec] = []
                for labels_path in labels_attrs["labels"]:
                    print("labels_path", labels_path)
                    ch.append(PlateLabels(self.group, labels_path=labels_path))
                return ch
        return []


class PlateLabels(Plate):
    def __init__(self, group: Group, labels_path: str):
        super().__init__(group)
        self.labels_path = labels_path

    def data(self) -> list[da.core.Array]:
        # return a dask pyramid...
        return get_pyramid_lazy(self.group, self.labels_path)

    def children(self) -> list[Spec]:
        # Need to override Plate.children()
        return []

    def metadata(self) -> dict:
        # override Plate metadata (no channel-axis etc)
        # TODO: read image-label metadata, colors etc
        return {
            "name": f"labels/{self.labels_path}",
        }


class Labels(Spec):
    @staticmethod
    def matches(group: Group) -> bool:
        return "labels" in Spec.get_attrs(group)

    # override to NOT yield self since node has no data
    def iter_nodes(self) -> Iterable[Spec]:
        attrs = Spec.get_attrs(self.group)
        for name in attrs["labels"]:
            g = self.group[name]
            if Label.matches(g):
                yield Label(g)


class Label(Multiscales):
    @staticmethod
    def matches(group: Group) -> bool:
        # label must also be Multiscales
        if not Multiscales.matches(group):
            return False
        return "image-label" in Spec.get_attrs(group)

    def add_parent_transform(
        self, transform: Dict[str, Any], parent_channel_axis: int | None
    ) -> None:
        # Add the parent transform to the current transform. If
        # parent_channel_axis is not in Label, we need to remove that axis
        # from the transform.
        label_channel_axis = self.metadata().get("channel_axis", None)
        if (
            parent_channel_axis is not None
            and parent_channel_axis != label_channel_axis
        ):
            print("remove axis from transform", parent_channel_axis)
            transform = remove_axis_from_transform(transform, parent_channel_axis)
        self.parent_transforms.append(transform)

    def metadata(self) -> Dict[str, Any]:
        # override Multiscales metadata
        # call super
        ms_data = super().metadata()
        print("Label metadata", ms_data)
        if ms_data is None:
            ms_data = {}
        if "channel_axis" in ms_data:
            ms_data.pop("channel_axis")

        attrs = Spec.get_attrs(self.group)
        image_label = attrs.get("image-label", {})
        colors: dict[int | bool, list[float]] = {}
        color_list = image_label.get("colors", [])
        if color_list:
            for color in color_list:
                try:
                    label_value = color["label-value"]
                    rgba = color.get("rgba", None)
                    if rgba:
                        rgba = [x / 255 for x in rgba]

                    if isinstance(label_value, (bool, int)):
                        colors[label_value] = rgba
                    else:
                        raise Exception("not bool or int")

                except Exception:
                    pass
                    # LOGGER.exception("invalid color - %s", color)

        props_list = image_label.get("properties", [])
        if props_list:
            props_by_labelid: dict[int, dict[str, str]] = {}
            for props in props_list:
                label_val = props["label-value"]
                props_by_labelid[label_val] = dict(props)
                del props_by_labelid[label_val]["label-value"]

            properties: Dict[str, List] = {}
            # First, create lists for all existing keys...
            for label_id, props_dict in props_by_labelid.items():
                for key in props_dict.keys():
                    properties[key] = []

            keys = list(properties.keys())

            properties["index"] = []
            for label_id, props_dict in props_by_labelid.items():
                properties["index"].append(label_id)
                # ...in case some objects don't have all the keys
                for key in keys:
                    properties[key].append(props_dict.get(key, None))
            ms_data["properties"] = properties

        return {
            "name": f"labels{self.group.name}",
            "colormap": colors,
            "visible": False,  # labels not visible initially
            **ms_data,
        }


def read_ome_zarr(root_group: Group) -> Callable:
    def f(*args: Any, **kwargs: Any) -> List[LayerData]:
        results: List[LayerData] = list()

        # # TODO: handle missing file
        # root_group = zarr.open(url)

        print("Root group", root_group.attrs.asdict())

        spec: Spec | None = None

        if Labels.matches(root_group):
            # Try starting at parent Image
            parent_path = root_group.store.root.parent
            parent_group = zarr.open_group(parent_path)
            if Multiscales.matches(parent_group):
                spec = Multiscales(parent_group)
            else:
                # not sure how to handle this?
                spec = Labels(root_group)
        elif Label.matches(root_group):
            # Try starting at parent Image - up 2 dirs
            parent_path = root_group.store.root.parent.parent
            parent_group = zarr.open_group(parent_path)
            if Multiscales.matches(parent_group):
                spec = Multiscales(parent_group)
            else:
                # not sure how to handle this?
                spec = Label(root_group)
        elif Bioformats2raw.matches(root_group):
            spec = Bioformats2raw(root_group)
        elif Multiscales.matches(root_group):
            spec = Multiscales(root_group)
        elif Plate.matches(root_group):
            spec = Plate(root_group)
        elif Scene.matches(root_group):
            print("Scene")
            spec = FusedScene(root_group)
        else:
            print("No matching spec", root_group)

        if spec:
            print("spec", spec)
            nodes = list(spec.iter_nodes())
            print("Nodes", nodes)
            for node in nodes:
                print("node", node, node.group)
                node_data = node.data()
                metadata = node.metadata()
                layer_type = "image"
                # print(Spec.get_attrs(node.group))
                if Label.matches(node.group) or isinstance(node, PlateLabels):
                    layer_type = "labels"
                rv: LayerData = (node_data, metadata, layer_type)
                results.append(rv)

        return results

    return f
