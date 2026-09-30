# Pose Pipeline

Loader-driven counterpart to the [programmatic Pose sample](../programmatic/README.md). The schema-v2 package under [`src/main/assets/pose-pipeline`](src/main/assets/pose-pipeline) contains two TFLite NPU models: a 128 x 128 pose detector and a 256 x 256 pose landmarker.

The `inference` graph acquires VST, detects and tracks the person ROI, runs landmark inference, and builds joint/bone transforms. The dependent `display` graph renders `pose_marker.gltf`. Both pipelines are loaded by the shared [`pipeline_common`](../../pipeline_common/README.md) host.

```sh
./gradlew :samples:pose_pipeline:installDebug
```

Launch **Pose Pipeline**, grant camera permission, and stand where the passthrough camera can see most of the body. Successful logs show both LiteRT models initialized and repeated `executePipeline ... err: 0` entries.
