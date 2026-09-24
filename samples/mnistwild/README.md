## Sample: MNIST Recognition

This sample builds a SpatialML application that recognizes hand-written digits that appear in the
VST camera feed. Unlike the other samples, both the preprocessing and inference pipelines are
described entirely in JSON ([mnist_pipeline.json](../../assets/mnistwild/mnist_pipeline.json)) and deserialized at runtime to create the
SpatialML pipeline graph.

The experience renders three virtual TV in front of the user. The cropped digit, its predicted class,
and the confidence score are displayed on the screen in real time.

![](../../assets/mnistwild/mnist_app.png)

### Assets

- Model: `mnist.serialized.bin` (generated from the classic MNIST network)
- Pipeline spec: `mnist_pipeline.json`
- Display mesh: `tv.gltf`

All assets are packaged under `assets/mnistwild` and loaded through Android's `AAssetManager`.

### Runtime Flow

1. **Framework setup**  
   `MnistWildApp` spins up a `FrameworkSession`, creates required global tensors, and loads the JSON
   pipeline specification plus GLTF assets on a background thread.

2. **Inference pipeline (JSON)**  
   The `mnist_pipeline.json` specification expands to the following operator chain:
   - `RECTIFIED_VST_ACCESS` acquires synchronized left/right RGB frames, timestamps, and camera parameters.
   - `GET_AFFINE` produces a perspective transform from hard-coded source/destination points so that the
     drawing area is cropped from the high-resolution VST frame.
   - `APPLY_AFFINE` crops the digit region (`crop_rgb_tensor`), after which
     `CONVERT_COLOR` converts it to grayscale.
   - Two `ASSIGNMENT` operators duplicate buffers for later use: one keeps the RGB crop for rendering,
     the other prepares a float copy.
   - `ARITHMETIC_COMPOSE` normalizes the grayscale image to `[0, 1]` by dividing by 255.
   - `RUN_MODEL_INFERENCE` feeds the normalized tensor into the MNIST model and produces
     `predicted_score` (confidence) and `predicted_class` (digit index).

   Tensors marked as placeholders in the JSON (`cropped_image`, `predicted_score`, `predicted_class`)
   are bound to global tensors so the app can read the results directly after each submission.

3. **Render pipeline (C++)**  
   A native pipeline (`CreateRenderPipeline`) maps the inference outputs onto the GLTF TV:
   - Updates dynamic textures so the RGB crop appears on the TV screen.
   - Draws the predicted class and score using `RenderCommand_DrawText`.
   - Positions three GLTF instances (digit text, score text, TV) via pose tensors to keep the
     visualization fixed in front of the user.

4. **Execution loop**  
   Two worker threads call `RunInferencePipeline` (~20 Hz) and `RunRenderPipeline` (~25 Hz) once all
   pipelines finish initializing.

### How to Build and Run

1. **Build and install**  
   Connect your PICO 4 Ultra device and run:
   ```bash
   ./gradlew :samples:mnistwild:installDebug
   ```

2. **Run the app**  
   Launch the "MnistWild" app from the PICO launcher. The app will request permissions for the VST camera; grant them to start the recognition pipeline.

### Customizing the JSON Pipeline

- Adjust `src_points` in the JSON if the digit capture area shifts in the physical world.
- Update `mnist.serialized.bin` for a retrained model; the binding names in the JSON must match the
  ONNX/TensorRT export (`input_1`, `_538`, `_539` in the default export).
- Additional preprocessing steps (e.g., blurring, adaptive thresholding) can be inserted by editing
  the JSON without touching the C++ side, as long as the placeholder tensors remain consistent.

Because the JSON is deserialized at runtime, developers can iterate on pipeline topology and operators
without recompiling the sample; copying an updated `mnist_pipeline.json` to the writable path is
enough to override the bundled asset.
