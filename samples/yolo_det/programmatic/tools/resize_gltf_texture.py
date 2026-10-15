#!/usr/bin/env python3
"""Create a lighter, wide label texture for the programmatic YOLO asset."""

import base64
import io
import json
from pathlib import Path
import sys

from PIL import Image


LABEL_TEXTURE_SIZE = (512, 128)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: resize_gltf_texture.py INPUT_GLTF OUTPUT_GLTF")

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    document = json.loads(input_path.read_text())

    buffer = bytearray(base64.b64decode(document["buffers"][0]["uri"].split(",", 1)[1]))
    image_view = document["images"][0]["bufferView"]
    view = document["bufferViews"][image_view]
    offset = view.get("byteOffset", 0)
    end = offset + view["byteLength"]
    if end < max(v.get("byteOffset", 0) + v["byteLength"] for v in document["bufferViews"]):
        raise RuntimeError("embedded label image is not the final buffer view")

    source = Image.open(io.BytesIO(buffer[offset:end])).convert("RGBA")
    target = source.resize(LABEL_TEXTURE_SIZE, Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    target.save(encoded, format="PNG", optimize=True)

    new_buffer = buffer[:offset] + encoded.getvalue()
    while len(new_buffer) % 4:
        new_buffer.append(0)
    view["byteLength"] = len(encoded.getvalue())
    document["buffers"][0]["byteLength"] = len(new_buffer)
    document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(new_buffer).decode("ascii")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, separators=(",", ":")))
    print(f"wrote {output_path} with {LABEL_TEXTURE_SIZE[0]}x{LABEL_TEXTURE_SIZE[1]} label texture")


if __name__ == "__main__":
    main()
