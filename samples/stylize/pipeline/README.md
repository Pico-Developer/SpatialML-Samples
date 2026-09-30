# Stylize Pipeline

Loader-driven counterpart to the [programmatic Stylize sample](../programmatic/README.md). The schema-v2 package under [`src/main/assets/stylize-pipeline`](src/main/assets/stylize-pipeline) contains the style-prediction and style-transfer TFLite models, both targeted at NPU, plus inference/display graphs and `portal.gltf`.

The native host keeps independent globals for the left and right controller paths, loads the five source style images from the shared `assets/stylize/textures` directory, and renders only the final stylized circles. Controller positions are projected into view space at a depth of 2 metres, with `view_locked` enabled for parity with the programmatic sample. No tensor readback is used.

```sh
./gradlew :samples:stylize_pipeline:installDebug
```

Move either controller to reposition its output circle. Controller button presses cycle that side's style image; the source artwork is used as model input but is not separately rendered.
