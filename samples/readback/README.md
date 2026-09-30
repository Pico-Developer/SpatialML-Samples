# Readback

This sample demonstrates readback of a SpatialML global tensor through CPU buffers or GPU textures. It acquires a VST camera frame and writes it as `output.png` in the app's external-files directory. It does not run an ML model.

## Variants

- `readback_cpu` uses `XrReadbackTensorBufferPICO` to copy CPU buffer data.
- `readback_vulkan` uses `XrReadbackTexturePICO` and Vulkan.
- `readback_opengl` uses `XrReadbackTexturePICO` and OpenGL ES.

The Vulkan and OpenGL flavors enable the corresponding CMake definitions automatically.

## Build and install

```sh
# CPU buffer
./gradlew :samples:readback:installReadback_cpuDebug

# Vulkan texture
./gradlew :samples:readback:installReadback_vulkanDebug

# OpenGL ES texture
./gradlew :samples:readback:installReadback_openglDebug
```

Launch **Readback** and grant camera permission. Each successful readback replaces the PNG at:

```text
/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.readback/files/output.png
```

Pull it from a specific connected headset with:

```sh
adb -s <device-serial> pull /sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.readback/files/output.png
```

## Implementation

- `cpp/readback_file.h` — controller and readback state
- `cpp/readback_file.cpp` — VST pipeline, CPU-buffer acquisition, and PNG export
- `cpp/readback_vulkan.cpp` — Vulkan texture copy and export
- `cpp/readback_opengl.cpp` — OpenGL ES framebuffer readback and export

The `vstOutputLeftUint8Global` tensor holds the latest left-eye RGB image at 512 x 512. CPU mode uses `XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO`. GPU modes use `XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO` with `XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO`.
