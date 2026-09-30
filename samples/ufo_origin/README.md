# UFO Face Tracking: Raw C API Reference

This legacy implementation produces the same face-tracking effect as [`samples/ufo`](../ufo/README.md), but calls the SpatialML OpenXR extension directly instead of using the helpers in [`external/SpatialML-XR-Utils/base/securemr_utils`](../../external/SpatialML-XR-Utils/base/securemr_utils).

It is retained as a low-level API reference and is not included by the current root `settings.gradle`. The actively maintained UFO module is `:samples:ufo`.

The sample performs these operations directly:

1. Creates the SpatialML framework, tensors, pipelines, and operators with the OpenXR C API.
2. Acquires stereo VST images and camera calibration.
3. Runs MediaPipe face detection.
4. Projects the selected face point into 3D.
5. Updates and renders the UFO glTF transform.

See the [programmatic UFO README](../ufo/README.md) for the current model, behavior, and build command.
