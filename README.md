# SpatialML Samples

## Samples for SpatialML.

| Face tracking | Pose estimation | YOLO |
|:-------------:|:---------------:|:----:|
| ![Face tracking Demo](docs/Demo-UFO.gif) | ![Pose estimation demo](docs/Demo-Pose.gif) | ![YOLO demo](docs/Demo-YOLO.gif) |
| **Mnistwild** | **Rubic's cube** | **Whack-a-Mole** |
| ![Mnistwild demo](docs/mnistwild.gif) | ![Rubics cube demo](docs/rubics_cube.gif) | ![Whack-a-Mole demo](docs/whackamole.gif)|
| **Stylize** | **MediaPipe Face Pipeline** |  |
| ![Stylize demo](docs/stylize.gif) | ![MediaPipe Face Pipeline demo](docs/face_mediapipe_pipeline.gif) |  |

## Nerual-Network (NN) Models Used in Samples

| Demo | Model | Source |
|------|-------|--------|
| UFO (Face Detection) | MediaPipe Face Detection | [Qualcomm AI Hub - MediaPipe Face Detection](https://aihub.qualcomm.com/models/mediapipe_face?searchTerm=face) |
| Face MediaPipe Pipeline | MediaPipe Face Detection | [Qualcomm AI Hub - MediaPipe Face Detection](https://aihub.qualcomm.com/models/mediapipe_face) |
| Pose (Pose Estimation) | MediaPipe Pose Landmarker | [Google AI Edge - Pose Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) |
| YOLO (Object Detection) | YOLOv11 Detection | [Qualcomm AI Hub - YOLOv11 Detection](https://aihub.qualcomm.com/models/yolov11_det?searchTerm=yolo) |
| MNIST (Digit Recognition) | Classic MNIST | [MNIST Database](https://en.wikipedia.org/wiki/MNIST_database) |
| Whack-a-Mole |  MediaPipe Pose Landmarker | [Google AI Edge - Pose Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) |
| Rubic's Cube (Solver) | No model used | N/A (Internal) |
| Stylize | Google arbitrary-image-stylization-v1 (TFLite) | [Kaggle - arbitrary-image-stylization-v1](https://www.kaggle.com/models/google/arbitrary-image-stylization-v1/tfLite) |

## SpatialML Pipeline Zoo

Ready-to-use SpatialML pipeline packages are available in the PICO [SpatialML Pipeline Zoo](https://huggingface.co/collections/picoxr/spatialml-pipeline-zoo) on Hugging Face. These packages bundle serialized models, JSON pipeline definitions, manifests, and related assets so apps can load complete SpatialML pipelines through the utility classes under `base/securemr_utils`.

See the [`face_mediapipe_pipeline`](samples/face_mediapipe_pipeline/README.md) sample for an app that downloads the [Face MediaPipe pipeline package](https://huggingface.co/picoxr/face-mediapipe-pipeline) during the Gradle build and uses `SecureMrUtils::LoadModelPackagePipelinesFromAssets(...)` to construct its detection and display pipelines automatically.

## Project aims

The projects demonstrates the functionalities and usage of
the SpatialML interfaces through several out-of-the-box
sample applications. The applications each achieve some
customized MR-based effects with deployment of open-sourced
machine learning algorithms. 

Additionally, the project provides a set of utility classes,
located under [`base/securemr_utils`](base/securemr_utils/README.md)
to simplify your
development of SpatialML-enabled applications.

The [`sample/mnist`](samples/mnistwild) also demonstrates how to defines the SpatialML
pipelines completely in JSON, using which allows your application
to dynamically load and update pipelines at run time, without hard
coding the algorithms into the app. 

A docker file together with necessary resources are also 
contained under the `Docker/` directory, if you would like
deploy your own algorithm packages. 


## Repository Structure

```
.
├── Docker
|                Docker files and resources to convert ML algorithm packages
├── assets
│   │            Asset required by each sample project
│   │
│   ├── UFO
│   │            Assets used by sample "ufo" and "ufo_origin"
│   └── common
│                Assets shared by all sample projects
│
├── base
│   │            Base source codes, shared by sample projects,
│   │            including the fundermental OpenXR codes
│   │
│   ├── oxr_utils
│   │            Utility for fundermental OpenXR APIs, such as
│   │            verification XR API results and vulkan renderer
│   │
│   ├── securemr_utils
│   │            Utility for SpatialML samples, to simplify the logic
│   │            in samples. Note, to demonstrate the raw usage of
│   │            the C-API for SpatialML provided as an OpenXR extension,
│   │            some sample projects are written by directly calling
│   │            the C-API instead of using the utility classes here. 
│   │
│   └── vulkan_shaders
|                Vulkan shaders for the client
|
├── docs
│                Documentations
|
├── external
|                External dependencies
|
├── samples
│   │            Directory for all sample projects. 
│   │         
│   ├── ufo
│   │             This is a sample showing a UFO "chasing" the human being
│   │             whoever it sees. The sample app uses an open-sourced
│   │             face detection model from MediaPipe.  
│   │
│   ├── mnistwild
│   │             Hand-written digit recognition using a self-trained MNIST-based
│   │             inference pipeline.
│   │
│   ├── readback
│   │             A minimumal demo that shows the usage of the readback APIs, which
│   │             allows an app, if proper camera or spatial-data permission(s)
│   │             is granted, to read the tensor content back from the SpatialML server.
│   │             This demo does not deploy any algorithms, nor present any render
│   │             effects: it simply calls the readback methods to obtain the camera
│   │             image and save it to local storage. 
│   │
│   ├── stylize
│   │             A stylization demo that applies an artistic filter to the camera feed,
│   │             showing a live, painterly effect over the scene.
│   │
│   ├── face_mediapipe_pipeline
│   │             A face detection and rendering demo that downloads a SpatialML
│   │             pipeline zoo package and builds the SpatialML pipelines from JSON
│   │             package metadata using the utility classes.
│   │
│   ├── rubics_cube
│   │             A Rubik's Cube scanner and solver with real-time
│   │             color classification, presenting the step-by-step instructions on
│   │             the screen.
│   │
│   ├── whackamole
│   │             A whack-a-mole MR game, which showcases how to run an NN-based
│   │             computer-vision (CV) algorithm together with customized JavaScript
│   │             for post-processing to achieve fancier MR effects, such as collision
│   │              detection, scoring and user interfaces. 
│   │
│   └── model_inspect
│                 Diagnostic tool for validating serialized models 
│                 on device.
|
└── ...
```

## Prerequisite

#### (A) To run the demo, you will need

1. A PICO 4 Ultra device with the latest system update (OS version >= 5.15.0)
1. Android Studio, with Android NDK installed, suggested NDK version = 25
1. Gradle and Android Gradle plugin (usually bundled with Android Studio install),
   suggested Gradle version = 8.7, Android Gradle Plugin version = 8.3.2
1. Java version at least 17 (required by the Android Gradle Plugin), recommended to be 21

#### (B) To run the docker, you will need

1. Docker desktop installed

## Deployment and Test

1. Install and configure according to the [prerequisite](#prerequisite). 
1. Open the repository root in Android Studio, as an Android project
1. After project sync, you will find several modules detected by the Android Studio, all under the `samples` folder: 
     1. `pose` which contains a pose detection demo
     1. `ufo` which contains a face detection demo
     1. `yolo` which contains an object detection demo
     1. `mnistwild` which contains a hand-written digit recognition demo
     1. `readback` which contains a minimal demo showing the usage of the readback APIs
     1. `stylize` which contains an artistic stylization demo for the camera feed
     1. `face_mediapipe_pipeline` which downloads a SpatialML pipeline zoo package and builds the SpatialML pipelines from JSON package metadata
     1. `whackamole` which contains a MR whack-a-mole game, based on pose detection and customized post-processing and game logic implemented in JavaScript
     1. `rubics_cube` which contains a Rubik's Cube solver demo
     1. `model_inspect` which is a utility for model validation
     1. `ufo-origin`, the same demo as `ufo`, but written using direct calls to the OpenXR C-API, with no 
         simplification using SpatialML utility classes.
1. Connect to a PICO 4 Ultra device with the latest OS update installed
1. Select the module you want to run, and click the launch button.

## PICO Developer Reference 

You can view the full SpatialML document via
[this link to PICO Developer website](https://developer-cn.picoxr.com/document/native/securemr-overview/). 
