# YOLO Detection Pipeline

JSON/package counterpart to the [programmatic YOLO sample](../programmatic/README.md). The schema-v2 package under [`src/main/assets/yolo-det-pipeline`](src/main/assets/yolo-det-pipeline) preserves the programmatic detector's current model, post-processing, temporal filtering, calibrated geometry, labels, and single-glTF renderer.

## Execution

The package contains three ordered pipelines:

1. `inference` reads the 640 x 640 rectified VST stereo pair, converts the first image from BGR `UINT8` to `FLOAT32`, and runs YOLOX-M on the Qualcomm NPU. A compact four-thread CPU LiteRT model decodes the 8,400 raw predictions and returns the top 128 candidates. Class-aware NMS uses IoU `0.5` and retains up to 10 results.
2. `labels` draws `class confidence%` into the 10 label textures. It runs on each of the first three submissions so the third-frame confirmed detection has text before it becomes visible, then every 15 submissions because text rendering is much more expensive than pose updates.
3. `display` applies 20 node-local poses to one shared glTF and renders it. The first 10 nodes are colored detection frames; the remaining 10 are their attached horizontal labels.

The native host submits at a 100 ms target interval. Inference and display run every submission. Label textures refresh on startup frames 0, 1, and 2, then every 15 submissions. The pipelines are chained with run handles, so text rendering completes before the display pipeline can expose a newly confirmed label and consumers do not race shared tensors.

## Detection behavior

- Display confidence threshold: `0.50`.
- Maximum visible detections: 10.
- NMS: class-aware, IoU `0.5`.
- Temporal confirmation: three consecutive same-class matches with IoU above `0.40`.
- Position: each box center is reconstructed by `UV_TO_3D_IN_CAM_SPACE` from the live stereo VST tensors.
- Invalid depth: a 2D detection is hidden when stereo reconstruction does not return a point in front of the camera, preventing near-plane clipping artifacts.
- Render depth: valid stereo points are visualized between 1.5 m and 5.0 m along the same camera ray. Box and label scale is recomputed at that depth, preserving the detected screen-space bounds while avoiding far-plane artifacts from unreliable very-large stereo depths.
- Box bounds: decoded corners are clipped to the 640 x 640 VST image and non-finite or degenerate boxes are discarded before UV reconstruction and rendering.
- Size: box and label scales use the decoded pixel dimensions, reconstructed depth, and VST focal lengths. There is no generic box-size multiplier.
- Labels: the same 80-class COCO ordering as the programmatic sample, confidence percentage included, per-slot frame/text colors, and an overlap fallback that moves a lower-confidence label from the top to the bottom of its box.
- Orientation: frames and labels remain on fixed world axes. Labels stay horizontal and attached to their boxes instead of inheriting headset roll.

The renderer uses [`frame2_multi_10_pipeline.gltf`](src/main/assets/yolo-det-pipeline/gltf/frame2_multi_10_pipeline.gltf), which contains all 10 frames and 10 labels. Keeping them in one glTF avoids the Vulkan descriptor-pool failures observed with many independent glTF instances. Every node is authored at zero scale so an uninitialized frame or label cannot appear at the world origin before `UPDATE_GLTF` supplies its first valid local pose.

## Regeneration and checks

The JSON is generated so repeated detection slots remain consistent:

```sh
node samples/yolo_det/pipeline/generate_pipeline_json.mjs
node samples/yolo_det/pipeline/test_geometry.mjs
./gradlew :samples:yolo_det_pipeline:assembleDebug
```

The generated package also requires byte-identical copies of these programmatic assets:

- `assets/yolo_det_programmatic/yolox_postprocess.tflite` -> `src/main/assets/yolo-det-pipeline/model/yolox_postprocess.tflite`
- `frame2_multi_10_pipeline.gltf` is generated from `assets/yolo_det_programmatic/frame2_multi_10.gltf` with the initial zero-scale node state described above.

The focused test checks model targets and tensor contracts, class-aware NMS, three-frame confirmation, camera-calibrated poses, label collision behavior, confidence text, the single-glTF layout, and asset hashes.

Install and launch the sample with:

```sh
./gradlew :samples:yolo_det_pipeline:installDebug
```

Successful runtime logs include package deserialization followed by repeated `OpenMR: executePipeline ... err: 0` entries. The YOLO model target is `npu`; the compact postprocessor and graph operators such as JavaScript/NMS run outside the NPU.
