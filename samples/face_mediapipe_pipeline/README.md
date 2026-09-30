# Face MediaPipe Pipeline

This sample loads a complete schema-v2 SpatialML package and renders a glTF frame over the highest-confidence face detected in the VST camera image. Pipeline tensors and operators are declared in JSON instead of being constructed individually in C++.

![Face MediaPipe Pipeline demo](face_mediapipe_pipeline.gif)

## Bundled package

The package is committed under [`src/main/assets/xr-face-mediapipe-pipeline`](src/main/assets/xr-face-mediapipe-pipeline). Gradle does not download or replace these assets. This keeps builds reproducible and allows package changes to be reviewed with the sample.

```text
xr-face-mediapipe-pipeline/
├── manifest.json
├── gltf/
│   └── frame.gltf
├── model/
│   └── face_detector.tflite
└── pipeline/
    ├── face_detection_pipeline.json
    └── face_display_pipeline.json
```

The package uses `schema_version: "2"`; model metadata is inline on the `RUN_MODEL_INFERENCE` operator. There is no separate `model.json`. The TFLite face detector targets NPU.

The corresponding published package is available from the [Face MediaPipe pipeline repository](https://huggingface.co/picoxr/xr-face-mediapipe-pipeline), and related packages are listed in the [SpatialML Pipeline Zoo](https://huggingface.co/collections/picoxr/spatialml-pipeline-zoo).

## Runtime flow

The native host calls `SecureMrUtils::LoadModelPackagePipelinesFromAssets(...)` with `xr-face-mediapipe-pipeline`, then submits:

1. `detection` — acquires the 580 x 326 VST frame, preprocesses it, runs MediaPipe Face Detection on LiteRT NPU, and produces `post_det`.
2. `display` — converts the detection into the face-frame transform, renders the confidence label, and draws `frame.gltf`.

The display submission depends on the detection run so it never consumes a partially updated result.

## Build and run

```sh
./gradlew :samples:face_mediapipe_pipeline:installDebug
```

Launch **Face MediaPipe Pipeline**, grant camera permission, and look toward a face. The frame and confidence text should remain aligned with the face as the headset or subject moves.

For detailed startup, NPU, and inference logs:

```sh
adb -s <device-serial> shell setprop debug.pico.securemr.debug 1
adb -s <device-serial> logcat | grep -E 'FaceMediaPipePipeline|OpenMR|Secure MR|litert'
```

Successful execution includes `OpenMR: createFramework ... err: 0`, LiteRT Qualcomm/NPU initialization, and repeated `OpenMR: executePipeline ... err: 0` messages. A camera `PREPROCESS_ERROR` occurs before package/model loading and indicates a device capture/VST problem rather than invalid package JSON.
