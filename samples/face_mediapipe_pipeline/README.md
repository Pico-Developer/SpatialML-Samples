# Sample: Face MediaPipe Pipeline

This sample demonstrates how to run a prebuilt SpatialML pipeline package from the PICO pipeline zoo. Instead of hard-coding every SpatialML tensor and operator in C++, the app downloads a package that contains the model, JSON pipeline definitions, manifest metadata, and glTF assets, then uses the SpatialML utility loaders to build and run the pipelines automatically.

![Face MediaPipe Pipeline demo](face_mediapipe_pipeline.gif)

## Pipeline zoo package

PICO publishes ready-to-use SpatialML pipeline packages in the pipeline zoo on Hugging Face:

- [SpatialML Pipeline Zoo collection](https://huggingface.co/collections/picoxr/spatialml-pipeline-zoo)
- [Face MediaPipe pipeline package](https://huggingface.co/picoxr/xr-face-mediapipe-pipeline)

The face detector model used by this package is based on [Qualcomm AI Hub - MediaPipe Face Detection](https://aihub.qualcomm.com/models/mediapipe_face).

Optionally, the Gradle build can download the Face MediaPipe package automatically from Hugging Face and unpack it into the app assets directory:

```
samples/face_mediapipe_pipeline/src/main/assets/xr-face-mediapipe-pipeline/
```

The downloaded package is intentionally ignored by git because it is fetched from Hugging Face during the build when needed.

If you are not using the Gradle download task, download the package zip from the [Face MediaPipe pipeline package](https://huggingface.co/picoxr/xr-face-mediapipe-pipeline), extract it, and place the extracted `xr-face-mediapipe-pipeline` folder under:

```
samples/face_mediapipe_pipeline/src/main/assets/
```

The native code points to this asset folder by relative path:

```cpp
constexpr const char* kModelPackageAssetPath = "xr-face-mediapipe-pipeline";
```

The package contains the files needed to describe and run the complete pipeline:

```
xr-face-mediapipe-pipeline/
├── manifest.json
├── gltf/
│   └── frame.gltf
├── model/
│   ├── face_detector.tflite
│   └── model.json
└── pipeline/
    ├── face_detection_pipeline.json
    └── face_display_pipeline.json
```

The `manifest.json` names the available pipelines, model files, and runtime metadata. The files under `pipeline/` define the SpatialML tensors and operators: the detection pipeline handles VST input, model inference, and post-processing; the display pipeline consumes the detection tensor, projects it into 3D, and renders the frame glTF overlay.

## How the sample uses the package

At runtime, the sample calls `SecureMrUtils::LoadModelPackagePipelinesFromAssets(...)` with the asset root `xr-face-mediapipe-pipeline`. The utility reads the package `manifest.json`, loads the referenced pipeline JSON files, patches model operators to use the packaged serialized model, creates shared global tensor bindings between the detection and display pipelines, and binds packaged glTF assets through pipeline placeholders.

The app then submits two pipelines in sequence:

1. `detection`: reads the VST camera tensors, runs MediaPipe face inference, and writes detection output.
2. `display`: consumes the detection output, projects it into camera/world space, and renders a glTF face-frame overlay.

## Build and run

Build or install the sample normally; if the package assets are not already present, Gradle can download the package before merging assets:

```
./gradlew :samples:face_mediapipe_pipeline:installDebug
```

Then launch **Face MediaPipe Pipeline** on a supported PICO device with SpatialML support.

When the sample is running, look toward a face in the VST camera view. The app should detect the face and render the packaged glTF face-frame overlay aligned with the detected face. As the face moves, the overlay should follow the face in real time, showing that the downloaded detection and display pipelines are both running successfully.
