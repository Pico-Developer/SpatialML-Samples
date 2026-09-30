# Model Inspect Tool

This legacy diagnostic sample runs one serialized SpatialML model on a PICO device, logs its tensor metadata, and reads its outputs back to files. It is present in the source tree but is not included by the current root `settings.gradle`; add the module locally before invoking its Gradle task.

## Inputs

Push a serialized model and its metadata to a device directory, then point the app at that directory:

```sh
adb -s <device-serial> push model.serialized.bin /data/local/tmp/my_model/
adb -s <device-serial> push model.json /data/local/tmp/my_model/
adb -s <device-serial> shell setprop debug.securemr.model_inspect.model_dir /data/local/tmp/my_model
```

An input file is optional; random input is generated when it is omitted:

```sh
adb -s <device-serial> push input.bin /data/local/tmp/my_model/input.bin
adb -s <device-serial> shell setprop debug.securemr.model_inspect.input /data/local/tmp/my_model/input.bin
```

After enabling the module, build and install it with:

```sh
./gradlew :samples:model_inspect:installDebug
```

Outputs are written as `model_inspect_output_<tensor>.bin` under:

```text
/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.model_inspect/files/model_inspect/
```

Logcat also prints each output's tensor shape, data type, and a short value preview. This tool targets the older serialized-model contract; current active samples use LiteRT/TFLite models directly.
