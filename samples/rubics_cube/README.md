## Sample: Rubic's Cube

This sample demonstrates a SpatialML-powered Rubik's Cube scanner and solver for PICO. It guides you through scanning each face via passthrough, classifies sticker colors on-device, computes a solution, and visualizes the turn sequence with an overlaid virtual cube and UI hints.

### Code walk-through

The orchestrator lives in `cpp/rubics_cube.cpp` and wires together the subsystems below:

- **Camera + readback**: `CreateRelaxMrReadbackPipeline` opens the SpatialML VST pipeline to produce stereo RGB tensors. `TensorReadback` continuously copies the left-eye tensor `vstOutputLeftUint8Global` to the CPU for color analysis.
- **Computer vision**: `CameraManager`, `ScanManager`, and `ColorClassifier` process readback frames to detect facelets while `VirtualCube` mirrors the detected state and `CubeModel` tracks geometry.
- **Solving + UI**: Once all faces are captured, `Solver` computes a solution. `ScanUi` renders overlay arrows/labels that guide scanning, and `VirtualCube` animates the resulting turn sequence.

Pipelines:

1. `m_secureMrVSTImagePipeline` — captures stereo passthrough RGB and exposes the left-eye tensor (`vstOutputLeftUint8Global`) for CPU-side processing, plus timestamp/camera calibration for pose alignment.
2. A lightweight readback loop (`RunRelaxMrReadbackPipeline`) uses `TensorReadback` to pull the left-eye image at ~20 Hz; all CV runs on the CPU, so no extra inference pipeline is required.

Rendering and UI run in the OpenXR layer using the custom overlay and virtual cube renderers, driven by the SpatialML camera/readback outputs.

### How to Build and Run

1. **Build and install**  
   Connect your PICO 4 Ultra device and run:
   ```bash
   ./gradlew :samples:rubics_cube:installDebug
   ```

2. **Run the app**  
   Launch the "Rubic's Cube" app. Follow the on-screen instructions to scan each face of your physical cube.

***
