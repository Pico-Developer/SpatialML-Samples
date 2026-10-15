#!/usr/bin/env python3
"""Generate one glTF containing independently movable frames and labels.

The template's five node transforms are baked into its vertex data, allowing
each detection frame and label to be represented by directly movable nodes.
Labels use normalized geometry and independent transforms, so they can keep a
readable angular size without changing the detector-derived frame dimensions.
Label materials and textures remain independent so RenderText can update each
label without allocating a separate Vulkan model.
"""

import base64
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_INSTANCE_COUNT = 10
DETECTION_COLORS = [
    (255, 90, 95, 255),
    (0, 212, 255, 255),
    (124, 252, 0, 255),
    (255, 215, 0, 255),
    (255, 77, 255, 255),
    (255, 140, 0, 255),
    (77, 124, 254, 255),
    (0, 230, 118, 255),
    (255, 111, 181, 255),
    (179, 136, 255, 255),
]


def node_matrix(node):
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def accessor_array(document, buffer, accessor_index):
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    if accessor["componentType"] != 5126 or accessor["type"] != "VEC3":
        raise RuntimeError("expected a tightly packed float32 VEC3 accessor")
    if view.get("byteStride") not in (None, 12):
        raise RuntimeError("strided position/normal accessors are not supported")
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    array = np.frombuffer(buffer, dtype="<f4", count=count * 3, offset=offset).reshape(count, 3)
    return accessor, array


def bake_node_transform(document, buffer, node, mesh):
    matrix = node_matrix(node)
    normal_matrix = np.linalg.inv(matrix[:3, :3]).T
    for primitive in mesh["primitives"]:
        position_accessor, positions = accessor_array(
            document, buffer, primitive["attributes"]["POSITION"]
        )
        transformed_positions = (
            np.concatenate((positions.copy(), np.ones((positions.shape[0], 1), dtype=np.float32)), axis=1)
            @ matrix.T
        )[:, :3]
        positions[:] = transformed_positions
        position_accessor["min"] = transformed_positions.min(axis=0).tolist()
        position_accessor["max"] = transformed_positions.max(axis=0).tolist()

        _, normals = accessor_array(document, buffer, primitive["attributes"]["NORMAL"])
        transformed_normals = normals.copy() @ normal_matrix.T
        lengths = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
        transformed_normals /= np.maximum(lengths, 1.0e-8)
        normals[:] = transformed_normals


def normalize_mesh_geometry(document, buffer, mesh):
    """Center a baked mesh and normalize its bounds to one unit per axis."""
    primitive = mesh["primitives"][0]
    position_accessor, positions = accessor_array(
        document, buffer, primitive["attributes"]["POSITION"]
    )
    _, normals = accessor_array(document, buffer, primitive["attributes"]["NORMAL"])

    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    size = maximum - minimum
    if np.any(size <= 0.0):
        raise RuntimeError("label geometry must have non-zero bounds")

    positions[:] = (positions - (minimum + maximum) * 0.5) / size
    position_accessor["min"] = positions.min(axis=0).tolist()
    position_accessor["max"] = positions.max(axis=0).tolist()

    transformed_normals = normals.copy() * size
    lengths = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
    transformed_normals /= np.maximum(lengths, 1.0e-8)
    normals[:] = transformed_normals


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "usage: generate_multi_detection_gltf.py INPUT_GLTF OUTPUT_GLTF [INSTANCE_COUNT]"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    instance_count = int(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_INSTANCE_COUNT
    if not 1 <= instance_count <= 100:
        raise ValueError("INSTANCE_COUNT must be between 1 and 100")

    document = json.loads(input_path.read_text())
    if len(document.get("scenes", [])) != 1 or len(document.get("nodes", [])) != 5:
        raise RuntimeError("expected the five-node frame2.gltf template")
    if len(document.get("meshes", [])) != 5 or len(document.get("materials", [])) != 2:
        raise RuntimeError("unexpected frame2.gltf mesh/material layout")
    if len(document.get("textures", [])) != 1 or len(document.get("images", [])) != 1:
        raise RuntimeError("unexpected frame2.gltf texture layout")

    template_nodes = deepcopy(document["nodes"])
    template_meshes = deepcopy(document["meshes"])
    frame_material = deepcopy(document["materials"][0])
    label_material_template = deepcopy(document["materials"][1])
    texture_template = deepcopy(document["textures"][0])
    image_template = deepcopy(document["images"][0])

    buffer_uri = document["buffers"][0]["uri"]
    if not buffer_uri.startswith("data:application/octet-stream;base64,"):
        raise RuntimeError("expected an embedded binary buffer")
    buffer = bytearray(base64.b64decode(buffer_uri.split(",", 1)[1]))
    for node in template_nodes:
        bake_node_transform(document, buffer, node, template_meshes[node["mesh"]])
    normalize_mesh_geometry(document, buffer, template_meshes[4])

    frame_nodes = []
    label_nodes = []
    meshes = []
    materials = []
    textures = []
    images = []

    for instance in range(instance_count):
        color = DETECTION_COLORS[instance % len(DETECTION_COLORS)]
        frame_material_instance = deepcopy(frame_material)
        frame_material_instance["name"] = f"DetectionFrameMaterial.{instance:02d}"
        frame_material_instance["pbrMetallicRoughness"]["baseColorFactor"] = [
            channel / 255.0 for channel in color
        ]
        frame_material_id = len(materials)
        materials.append(frame_material_instance)

        label_material = deepcopy(label_material_template)
        label_material["name"] = f"DetectionLabelMaterial.{instance:02d}"
        label_material["pbrMetallicRoughness"]["baseColorTexture"]["index"] = instance
        label_material_id = len(materials)
        materials.append(label_material)

        texture = deepcopy(texture_template)
        texture["source"] = instance
        textures.append(texture)

        image = deepcopy(image_template)
        image["name"] = f"DetectionLabelImage.{instance:02d}"
        images.append(image)

        frame_mesh_id = len(meshes)
        frame_primitives = []
        for mesh in template_meshes[:4]:
            primitive = deepcopy(mesh["primitives"][0])
            primitive["material"] = frame_material_id
            frame_primitives.append(primitive)
        meshes.append(
            {"name": f"DetectionFrameMesh.{instance:02d}", "primitives": frame_primitives}
        )
        frame_nodes.append(
            {"name": f"DetectionFrame.{instance:02d}", "mesh": frame_mesh_id}
        )

        label_primitive = deepcopy(template_meshes[4]["primitives"][0])
        label_primitive["material"] = label_material_id
        label_mesh_id = len(meshes)
        meshes.append(
            {
                "name": f"DetectionLabelMesh.{instance:02d}",
                "primitives": [label_primitive],
            }
        )
        label_nodes.append(
            {"name": f"DetectionLabel.{instance:02d}", "mesh": label_mesh_id}
        )

    nodes = frame_nodes + label_nodes
    document["nodes"] = nodes
    document["meshes"] = meshes
    document["materials"] = materials
    document["textures"] = textures
    document["images"] = images
    document["scenes"][0]["nodes"] = list(range(instance_count * 2))
    document["scenes"][0]["name"] = f"YOLO detections ({instance_count})"
    document["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(buffer).decode("ascii")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, separators=(",", ":")))
    print(
        f"wrote {output_path}: instances={instance_count}, roots={list(range(instance_count * 2))}, "
        f"nodes={len(nodes)}, meshes={len(meshes)}, materials={len(materials)}, textures={len(textures)}"
    )


if __name__ == "__main__":
    main()
