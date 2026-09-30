# YOLO Detection Pipeline

Loader-driven counterpart to the [programmatic YOLO sample](../programmatic/README.md). SpatialML graphs are declared in the schema-v2 package under [`src/main/assets/yolo-det-pipeline`](src/main/assets/yolo-det-pipeline). Its YOLO TFLite model targets NPU; the native host loads the package and submits its pipelines in order.

The inference graph also computes the frame's world pose. It reconstructs the selected box center with `UV_TO_3D_IN_CAM_SPACE`, moves near points along their camera ray to a minimum depth of 1.5 m, applies the programmatic sample's Y flip and 0.1 m X offset, and multiplies by the left camera-to-world transform at the captured timestamp. Frame scale is `[width / 2000, height / 2000, 0.05]` for the shared `frame2.gltf` asset. The same Y cross-axis cleanup as programmatic `RenderText` is applied after the matrix multiplication. The display graph consumes this world pose directly and shows boxes above confidence 0.6.

Camera slot names follow the existing programmatic sample for parity: its `cameraAccess` call passes the variable named `left` to slot 0, although the API calls that slot `right image`. YOLO and stereo reconstruction both use that same slot convention here. Changing eye names alone would change the image used by the model and the stereo input pairing.

The package displays the single highest-scoring prediction with its COCO class name above the box. The display graph maps `detection[5]` to the same 80 class names as `programmatic/cpp/coco_classes.h`, writes a space-padded 16-byte INT8 text tensor, and renders it onto the frame. Confidence still controls visibility. The programmatic sample uses NMS and up to three class-labelled boxes.

Text rendering uses a 1024 x 1024 canvas to match the texture embedded in `frame2.gltf`; a different canvas size is rejected by the renderer.

Run the focused geometry and binding checks from the repository root:

```sh
node samples/yolo_det/pipeline/test_geometry.mjs
./gradlew :samples:yolo_det_pipeline:assembleDebug
```

Install and launch the sample with:

```sh
./gradlew :samples:yolo_det_pipeline:installDebug
```

Successful runtime logs include `RUN_MODEL_INFERENCE` completion and repeated `OpenMR: executePipeline ... err: 0` entries.

