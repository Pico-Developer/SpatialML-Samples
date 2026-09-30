# MNIST Wild

This sample builds a SpatialML application that recognizes handwritten digits in the VST camera feed.
The preprocessing, LiteRT inference, and postprocessing graph is described in JSON
([mnist_pipeline.json](../../../assets/mnistwild/mnist_pipeline.json)) and deserialized at runtime.

Three virtual TVs display the cropped camera region, predicted digit, and confidence in real time.

![](../../../assets/mnistwild/mnist_app.png)

### Assets

- Model: `mnist.tflite`, converted from the MIT-licensed
  [ONNX Model Zoo `mnist-12.onnx`](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist) model
- Pipeline spec: `mnist_pipeline.json`
- Display mesh: `tv.gltf`

All runtime assets are packaged under `assets/mnistwild` and loaded through Android's `AAssetManager`.

### Runtime flow

1. `RECTIFIED_VST_ACCESS` acquires the left and right VST camera frames.
2. A 224 x 224 affine crop is retained for display on the virtual TV.
3. A second affine transform resizes the same camera region directly to the model's 28 x 28 input.
4. The inference image is converted to grayscale and normalized to `[0, 1]`.
5. `RUN_MODEL_INFERENCE` runs `mnist.tflite` on LiteRT NPU and produces ten digit logits.
6. A small JavaScript postprocessing operator applies a numerically stable softmax, selects the winning digit,
   and publishes its confidence.
7. The native render pipeline displays the crop, predicted class, and confidence on three GLTF instances.

### Build and run

Connect a PICO Space Pro and run:

```bash
./gradlew :samples:mnistwild:installDebug
```

Launch **MNIST Wild** from the PICO launcher and grant camera permission. Hold a dark handwritten digit
on a light background in the fixed crop region shown on the image TV.

### Customize the JSON pipeline

- Adjust `src_points` if the physical capture region moves.
- A replacement model must accept float32 NHWC input `[1, 28, 28, 1]` named `inputs_0` and emit
  float32 logits `[1, 10]` named `Identity`, or the JSON bindings must be updated to match.
- Additional preprocessing or postprocessing can be added in JSON without changing the C++ render pipeline,
  as long as the `cropped_image`, `predicted_score`, and `predicted_class` placeholders remain compatible.

The app first checks its writable data directory for `mnist_pipeline.json`, then falls back to the bundled asset.
