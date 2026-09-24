# Sample: Readback

This sample demonstrates SpatialML readback on global tensor by both CPU buffers and GPU textures (Vulkan / OpenGL ES). It acquires the VST camera frame via a SpatialML pipeline and exports a PNG file to the app’s external storage for quick verification.

## Expected Behavior
- After camera permission is granted, the app initializes the SpatialML pipeline.
- On each successful readback, a `output.png` is written to external storage.
- Exports PNG in external storage.
  
## Variants
- readback_cpu: uses XrReadbackTensorBufferPICO to read back CPU buffer data
- readback_vulkan: uses XrReadbackTexturePICO and Vulkan 
- readback_opengl: uses XrReadbackTexturePICO and OpenGL ES

## Build & Install
- CPU buffer:

```
./gradlew :samples:readback:installReadback_cpuDebug
```

- Vulkan texture:

```
./gradlew :samples:readback:installReadback_vulkanDebug
```

- OpenGL ES texture:

```
./gradlew :samples:readback:installReadback_openglDebug
```

The Gradle flavors configure CMake flags automatically:
- readback_vulkan: `-DREADBACK_USE_GPU=1 -DREADBACK_USE_VULKAN=1`
- readback_opengl: `-DREADBACK_USE_GPU=1 -DREADBACK_USE_OPENGL=1`

## Run
- Launch the installed app on the device.
- Grant camera permission when prompted.
- After the pipeline initializes, the app saves a PNG to external storage.

## Output
- The exported file is written to the app’s external files directory:
  - Example: `/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.readback/files/output.png`

## Implementation Notes
- Core controller: `readback.h`
- Sample logic and pipeline setup: `readback_file.cpp`
- OpenGL ES export: `readback_opengl.cpp`
- Vulkan export: `readback_vulkan.cpp`

### Key Tensor: vstOutputLeftUint8Global
- Global tensor holding the latest left‑eye VST RGB frame.
- Dimensions: 512×512, Channels: 3.  
  CPU mode: `XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO`.  
  GPU mode: usage `XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO`, data type `XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO`.

## Code Walk-through
- Controller initialization
  - In the `readback.cpp` constructor, readback-related OpenXR PICO extension function pointers are fetched via `FrameworkSession` (e.g., `xrCreateBufferFromGlobalTensorAsyncPICO`, `xrCreateTextureFromGlobalTensorAsyncPICO`).
  - Graphics context is initialized in `readback_vulkan.cpp` or `readback_opengl.cpp` depending on the build backend.
- CPU buffer readback
  - Issue: `RequestReadbackBuffer(...)`
  - Acquire: `TryAcquireReadbackBuffer(...)` uses `xrCreateBufferFromGlobalTensorCompletePICO` in two call idiom (query size, then allocate and read).
  - Export: `OutputReadbackBufferToFile(...)` writes RGB data using `stbi_write_png`.
- GPU texture readback
  - Issue: `RequestReadbackTexture(...)`
  - Acquire: `TryAcquireReadbackTexture(...)` returns an `XrReadbackTexturePICO`.
  - Export:
    - Vulkan: `OutputReadbackTextureToPath(...)` copies from `VkImage` to a staging buffer and saves RGBA PNG.
    - OpenGL ES: `OutputReadbackTextureToPath(...)` attaches texture to an FBO, `glReadPixels` RGBA, then saves PNG.
