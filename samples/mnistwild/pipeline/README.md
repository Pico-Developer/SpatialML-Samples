# MNIST Wild Pipeline

Loader-driven counterpart to the [programmatic MNIST sample](../programmatic/README.md). The complete schema-v2 package is committed under [`src/main/assets/mnistwild-pipeline`](src/main/assets/mnistwild-pipeline).

The `inference` pipeline captures a camera crop, converts it to 28 x 28 grayscale FP32, runs `mnist.tflite` on LiteRT NPU, and applies JavaScript softmax/class selection. The `display` pipeline consumes those tensors directly and renders the crop, predicted digit, and confidence on three virtual TVs.

```sh
./gradlew :samples:mnistwild_pipeline:installDebug
```

Launch **MNIST Wild Pipeline** and place a dark handwritten digit on a light background inside the crop region. Successful logs include LiteRT model initialization and repeated pipeline executions with `err: 0`.
