# UFO Face Tracking

This sample uses SpatialML face detection to move a disc-shaped UFO toward a visible person and keep it floating above the detected face.

The sample deploys the open-source MediaPipe face detector
through LiteRT NPU and renders a UFO in
glTF 2.0 format. 

![Demo for face tracking with an UFO chasing the detected face](../../docs/Demo-UFO.gif)

### Code walk-through

The sample loads the algorithm from `assets/UFO/face_detector.tflite`, which is
the LiteRT/TFLite MediaPipe face detection model. The UFO asset to be
rendered is `assets/UFO/UFO.gltf`, loaded to the global tensor `gltfAsset`.

There are four pipelines in this sample:

1. `m_secureMrVSTImagePipeline` to obtain the RGB images. Although we only use the left-eye 
    image for face detection, we still need the stereo-view RGB outputs (in global tensor
    `vstOutputLeftUint8Global` and `vstOutputRightUint8Global` respectively), the image timestamp
    (`vstTimestampGlobal`) and the camera matrix `vstCameraMatrixGlobal` for the 2D-to-3D pipeline
    to compute an accurate 3D position. 
2. `m_secureMrModelInferencePipeline`, as the key pipeline, to run the face detection
    algorithm on the RGB image from the above pipeline, and outputs the 2D coordinate
    of the face center, in the global tensor: `uvGlobal`.
3. `m_secureMrMap2dTo3dPipeline` uses the on-device depth sensor, to project back the
    2D coordinate of the face center and obtain the 3D coordinate under OpenXR's 
    XR_LOCAL reference space. The result 3D coordinate is stored in `currentPositionGlobal`.
4. `m_secureMrRenderingPipeline` updates the world pose of the "UFO" according to the detected
    coordinate in `currentPositionGlobal`. The pipeline also incorporates a damping algorithm
    to smooth the trajectory of UFO and mimicking the physics. The damped coordinate is
    updated to `previousPositionGlobal` for usage in the next frame. 

### Build and run

```sh
./gradlew :samples:ufo:installDebug
```

Launch **UFO Face Tracking (Refactored)** and grant camera permission. The UFO should move toward the detected face and remain positioned above it. Successful runtime logs show LiteRT NPU initialization and repeated pipeline submissions with `err: 0`.
