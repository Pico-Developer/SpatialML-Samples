#include "stylize_demo.h"
#include <sstream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <openxr/openxr.h>

#define STB_IMAGE_IMPLEMENTATION
#include <image_utils/stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <image_utils/stb_image_resize2.h>

extern AAssetManager* g_assetManager;

#define STYLE_IMAGE_DIM 256
#define CONTENT_IMAGE_DIM 384
#define VST_WIDTH 2048
#define VST_HEIGHT 1536
#define PREDICTED_STYLE_CHANNEL_SIZE 100

namespace SecureMR {

std::vector<char> LoadModelData(const char* modelPath) {
  if (!g_assetManager) {
    Log::Write(Log::Level::Error, "Asset manager is null");
    return {};
  }

  AAsset* const asset = AAssetManager_open(g_assetManager, modelPath, AASSET_MODE_BUFFER);
  if (!asset) {
    Log::Write(Log::Level::Error, Fmt("Failed to open asset: %s", modelPath));
    return {};
  }

  const off_t assetLength = AAsset_getLength(asset);
  std::vector<char> buffer(assetLength);
  
  const size_t bytesRead = AAsset_read(asset, buffer.data(), assetLength);
  AAsset_close(asset);

  if (bytesRead != static_cast<size_t>(assetLength)) {
    Log::Write(Log::Level::Error, Fmt("Failed to read asset: %s (read %zu of %jd bytes)", modelPath, bytesRead, static_cast<intmax_t>(assetLength)));
    return {};
  }

  return buffer;
}

// Load and resize image from assets to specified dimensions
std::vector<uint8_t> LoadAndResizeImage(const char* imagePath, int targetWidth, int targetHeight) {
  if (!g_assetManager) {
    Log::Write(Log::Level::Error, "Asset manager is null");
    return {};
  }

  // Load image data from assets
  AAsset* const asset = AAssetManager_open(g_assetManager, imagePath, AASSET_MODE_BUFFER);
  if (!asset) {
    Log::Write(Log::Level::Warning, Fmt("Failed to open image asset: %s", imagePath));
    return {};
  }

  const off_t assetLength = AAsset_getLength(asset);
  std::vector<uint8_t> assetData(assetLength);
  const size_t bytesRead = AAsset_read(asset, assetData.data(), assetLength);
  AAsset_close(asset);

  if (bytesRead != static_cast<size_t>(assetLength)) {
    Log::Write(Log::Level::Error, Fmt("Failed to read complete image asset: %s", imagePath));
    return {};
  }

  // Decode image using stb_image
  int width, height, channels;
  unsigned char* imageData = stbi_load_from_memory(assetData.data(), assetLength, &width, &height, &channels, 3); // Force RGB
  
  if (!imageData) {
    Log::Write(Log::Level::Error, Fmt("Failed to decode image: %s", imagePath));
    return {};
  }

  std::vector<uint8_t> result;
  
  // Resize if necessary
  if (width != targetWidth || height != targetHeight) {
    result.resize(targetWidth * targetHeight * 3);
    unsigned char* resizeResult = stbir_resize_uint8_srgb(imageData, width, height, width * 3, result.data(), targetWidth, targetHeight, targetWidth * 3, STBIR_RGB);
     if (!resizeResult) {
      Log::Write(Log::Level::Error, Fmt("Failed to resize image: %s", imagePath));
      stbi_image_free(imageData);
      return {};
    }
  } else {
    // Copy data directly if no resizing needed
    result.assign(imageData, imageData + (width * height * 3));
  }

  stbi_image_free(imageData);
  Log::Write(Log::Level::Info, Fmt("Successfully loaded and resized image: %s (%dx%d -> %dx%d)", 
                                   imagePath, width, height, targetWidth, targetHeight));
  return result;
}

StylizeDemo::StylizeDemo(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

StylizeDemo::~StylizeDemo() {
  keepRunning = false;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) runner.join();
  }
}

void StylizeDemo::CreateFramework() {
  Log::Write(Log::Level::Info, "Creating Framework...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, VST_WIDTH, VST_HEIGHT);
  Log::Write(Log::Level::Info, "Framework created.");
}

void StylizeDemo::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    CreateGlobalTensor();
    CreateSecureMrVSTImagePipeline();
    CreateSecureMrStylePredictionPipeline();
    CreateSecureMrStyleTransferPipeline();
    CreateSecureMrRenderingPipeline();

    {
      std::lock_guard<std::mutex> guard(initialized_mtx);
      globalTensorsInitialized = true;
      pipelineAllInitialized = true;
    }
    initialized.notify_all();
  });
}

void StylizeDemo::LoadStyleTextureFromAssetPath(const char* styleTexturePath, std::shared_ptr<GlobalTensor> targetTensor) {
  std::vector<uint8_t> styleTextureData = LoadAndResizeImage(styleTexturePath, STYLE_IMAGE_DIM, STYLE_IMAGE_DIM);
  if (!styleTextureData.empty()) {
    // Copy the loaded and resized image data to the target global tensor
    targetTensor->setData(reinterpret_cast<int8_t*>(styleTextureData.data()), styleTextureData.size());
    Log::Write(Log::Level::Info, Fmt("Style texture loaded successfully: %zu bytes", styleTextureData.size()));
  } else {
    Log::Write(Log::Level::Warning, "Failed to load style texture, using default pattern");

    // Create a simple fallback pattern if image loading fails
    std::vector<uint8_t> fallbackData(STYLE_IMAGE_DIM * STYLE_IMAGE_DIM * 3);
    for (int i = 0; i < STYLE_IMAGE_DIM * STYLE_IMAGE_DIM * 3; i += 3) {
      fallbackData[i + 0] = 128; // Red
      fallbackData[i + 1] = 64;  // Green
      fallbackData[i + 2] = 192; // Blue
    }
    targetTensor->setData(reinterpret_cast<int8_t*>(fallbackData.data()), fallbackData.size());

  }
}

void StylizeDemo::CreateGlobalTensor() {

  vstOutputLeftUint8Global = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {VST_WIDTH, VST_HEIGHT}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});

  // Create tensor for style textures
  for (size_t i = 0; i < Side::Count; ++i) {
    styleTextureGlobal[i] = std::make_shared<GlobalTensor>(
        frameworkSession,
        TensorAttribute{.dimensions = {STYLE_IMAGE_DIM, STYLE_IMAGE_DIM}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  }

  // Load initial style textures from assets
  for (size_t i = 0; i < Side::Count; ++i) {
    LoadStyleTextureFromAssetPath(STYLE_TEXTURE_PATHS[styleTextureIndex[i]], styleTextureGlobal[i]);
  }

  // Create tensor for predicted style features
  for (size_t i = 0; i < Side::Count; ++i) {
    predictedStyleGlobal[i] = std::make_shared<GlobalTensor>(
        frameworkSession,
        TensorAttribute{.dimensions = {1, 1}, .channels = PREDICTED_STYLE_CHANNEL_SIZE, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  }

  // Create dynamic texture tensor version of stylized output image
  for (size_t i = 0; i < Side::Count; ++i) {
    stylizedImageGPUGlobal[i] = std::make_shared<GlobalTensor>(
        frameworkSession,
        TensorAttribute{.dimensions = {CONTENT_IMAGE_DIM, CONTENT_IMAGE_DIM}, .channels = 3, .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO,
                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO});
  }


  // Create global tensor for portal pose
  for (size_t i = 0; i < Side::Count; ++i) {
    portalPoseGlobal[i] = std::make_shared<GlobalTensor>(
        frameworkSession,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  }

  // Create global tensor for source points
  for (size_t i = 0; i < Side::Count; ++i) {
    srcPointsGlobal[i] = std::make_shared<GlobalTensor>(
        frameworkSession,
        TensorAttribute{.dimensions = {3, 1}, .channels = 2, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  }

  // Load and create TV display model
  // TV display code removed as requested.

  std::vector<char> portalData = LoadModelData(PORTAL_GLTF_PATH);
  if (!portalData.empty()) {
    for (size_t i = 0; i < Side::Count; ++i) {
      portalGltf[i] = std::make_shared<GlobalTensor>(frameworkSession, portalData.data(), portalData.size());
    }
    const auto initPipelineGPU = std::make_shared<Pipeline>(frameworkSession);

    // Initialize portal position and orientation
    float portalPoseDataGPU[] = {
        1.0f, 0.0f, 0.0f, 1.0f,  // Position 1 meter to the right
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, -2.0f,  // Position 2 meters in front
        0.0f, 0.0f, 0.0f, 1.0f
    };

    const auto initPoseGPU = std::make_shared<PipelineTensor>(
        initPipelineGPU,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
        reinterpret_cast<int8_t*>(portalPoseDataGPU), sizeof(portalPoseDataGPU));

    const auto textureIdGPU = std::make_shared<PipelineTensor>(
        initPipelineGPU,
        TensorAttribute_ScalarArray{.dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});

    std::array<std::shared_ptr<PipelineTensor>, Side::Count> transitPlaceholders;
    std::array<std::shared_ptr<PipelineTensor>, Side::Count> texturePlaceholders;

    for (size_t i = 0; i < Side::Count; ++i) {
      portalPoseGlobal[i]->setData(reinterpret_cast<int8_t*>(portalPoseDataGPU), sizeof(portalPoseDataGPU));
      transitPlaceholders[i] = PipelineTensor::PipelineGLTFPlaceholder(initPipelineGPU);
      texturePlaceholders[i] = PipelineTensor::PipelinePlaceholderLike(initPipelineGPU, stylizedImageGPUGlobal[i]);
      initPipelineGPU->newTextureToGLTF(transitPlaceholders[i], texturePlaceholders[i], textureIdGPU);
      initPipelineGPU->execRenderCommand(std::make_shared<RenderCommand_UpdateMaterial>(
          transitPlaceholders[i],
          std::vector<uint16_t>{1u},
          RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR,
          textureIdGPU));
      initPipelineGPU->execRenderCommand(std::make_shared<RenderCommand_Render>(transitPlaceholders[i], initPoseGPU, true));
    }

    std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> initBindings{
        {transitPlaceholders[Side::Left], portalGltf[Side::Left]},
        {transitPlaceholders[Side::Right], portalGltf[Side::Right]},
        {texturePlaceholders[Side::Left], stylizedImageGPUGlobal[Side::Left]},
        {texturePlaceholders[Side::Right], stylizedImageGPUGlobal[Side::Right]}};
    initPipelineGPU->submit(initBindings, XR_NULL_HANDLE, nullptr);

  } else {
    Log::Write(Log::Level::Error, "Failed to load portal glTF model.");
  }

}

void StylizeDemo::CreateSecureMrVSTImagePipeline() {
  Log::Write(Log::Level::Info, "Creating VST Image Pipeline");

  m_secureMrVSTImagePipeline = std::make_shared<Pipeline>(frameworkSession);

  vstOutputLeftUint8VstPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftUint8Global);

  auto vstTimestampLocal = std::make_shared<PipelineTensor>(m_secureMrVSTImagePipeline, TensorAttribute_TimeStamp{});

  auto vstCameraMatrixLocal = std::make_shared<PipelineTensor>(
      m_secureMrVSTImagePipeline,
      TensorAttribute{.dimensions = {3, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  (*m_secureMrVSTImagePipeline)
      .cameraAccess(vstOutputLeftUint8VstPlaceholder, nullptr, vstTimestampLocal, vstCameraMatrixLocal);
}

void StylizeDemo::CreateSecureMrStylePredictionPipeline() {
  Log::Write(Log::Level::Info, "Creating Style Prediction Pipeline");
  
  m_secureMrStylePredictionPipeline = std::make_shared<Pipeline>(frameworkSession);
  
  // Load style predictor model
  std::vector<char> predictorModelData = LoadModelData(STYLE_PREDICTOR_MODEL_PATH);
  if (predictorModelData.empty()) {
    Log::Write(Log::Level::Error, "Failed to load style predictor model");
    return;
  }
  
  styleTexturePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrStylePredictionPipeline, styleTextureGlobal[Side::Left]);
  auto styleTextureFp32Local = std::make_shared<PipelineTensor>(
      m_secureMrStylePredictionPipeline,
      TensorAttribute{.dimensions = {STYLE_IMAGE_DIM, STYLE_IMAGE_DIM}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  predictedStyleOutputPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrStylePredictionPipeline, predictedStyleGlobal[Side::Left]);
  

  (*m_secureMrStylePredictionPipeline)
      .assignment(styleTexturePlaceholder, styleTextureFp32Local)
      .arithmetic("({0} / 255.0)", {styleTextureFp32Local}, styleTextureFp32Local)
      .runAlgorithm(predictorModelData.data(), predictorModelData.size(), {{"style_image", styleTextureFp32Local}},
                    {}, {{"mobilenet_conv/Conv/BiasAdd", predictedStyleOutputPlaceholder}}, {}, "style_predictor",
                    XR_SECURE_MR_MODEL_TYPE_LITE_RT_MODEL_PICO, XR_SECURE_MR_MODEL_TARGET_NPU_PICO);

}

void StylizeDemo::CreateSecureMrStyleTransferPipeline() {
  Log::Write(Log::Level::Info, "Creating Style Transfer Pipeline");
  
  m_secureMrStyleTransferPipeline = std::make_shared<Pipeline>(frameworkSession);
  
  // Load style transfer model
  std::vector<char> transferModelData = LoadModelData(STYLE_TRANSFER_MODEL_PATH);
  if (transferModelData.empty()) {
    Log::Write(Log::Level::Error, "Failed to load style transfer model");
    return;
  }

  auto slicedVSTLocal = std::make_shared<PipelineTensor>(
      m_secureMrStyleTransferPipeline,
      TensorAttribute{.dimensions = {CONTENT_IMAGE_DIM, CONTENT_IMAGE_DIM}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});

  auto vstOutputLeftFp32Local = std::make_shared<PipelineTensor>(
      m_secureMrStyleTransferPipeline,
      TensorAttribute{.dimensions = {CONTENT_IMAGE_DIM, CONTENT_IMAGE_DIM}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto stylizedImageLocal = std::make_shared<PipelineTensor>(
      m_secureMrStyleTransferPipeline,
      TensorAttribute{.dimensions = {CONTENT_IMAGE_DIM, CONTENT_IMAGE_DIM}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  vstOutputLeftUint8StylePlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStyleTransferPipeline, vstOutputLeftUint8Global);
  predictedStyleInputPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrStyleTransferPipeline, predictedStyleGlobal[Side::Left]);
  stylizedImageOutputPlaceholderGPU = PipelineTensor::PipelinePlaceholderLike(m_secureMrStyleTransferPipeline, stylizedImageGPUGlobal[Side::Left]);
  
  srcPointsPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrStyleTransferPipeline, srcPointsGlobal[Side::Left]);

  auto affineTransformLocal = std::make_shared<PipelineTensor>(
      m_secureMrStyleTransferPipeline,
      TensorAttribute{.dimensions = {2, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  std::array<float, 6> dstPoints = {
      0.0f, 0.0f,
      (float)CONTENT_IMAGE_DIM, 0.0f,
      0.0f, (float)CONTENT_IMAGE_DIM
  };

  (*m_secureMrStyleTransferPipeline)
      .getAffine(srcPointsPlaceholder, dstPoints, affineTransformLocal)
      .applyAffine(affineTransformLocal, vstOutputLeftUint8StylePlaceholder, slicedVSTLocal)
      .assignment(slicedVSTLocal, vstOutputLeftFp32Local)
      .arithmetic("({0} / 255.0)", {vstOutputLeftFp32Local}, vstOutputLeftFp32Local)
      .runAlgorithm(transferModelData.data(), transferModelData.size(), {{"content_image", vstOutputLeftFp32Local},
        {"mobilenet_conv/Conv/BiasAdd", predictedStyleInputPlaceholder}}, {},
        {{"transformer/expand/conv3/conv/Sigmoid", stylizedImageLocal}}, {}, "style_transfer",
        XR_SECURE_MR_MODEL_TYPE_LITE_RT_MODEL_PICO, XR_SECURE_MR_MODEL_TARGET_NPU_PICO)
      .arithmetic("({0} * 255.0)", {stylizedImageLocal}, stylizedImageLocal)
      .assignment(stylizedImageLocal, stylizedImageOutputPlaceholderGPU);
      
}

void StylizeDemo::CreateSecureMrRenderingPipeline() {
  Log::Write(Log::Level::Info, "Creating Rendering Pipeline");
  
  m_secureMrRenderingPipeline = std::make_shared<Pipeline>(frameworkSession);
  
  for (size_t i = 0; i < Side::Count; ++i) {
    portalGltfPlaceholder[i] = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, portalGltf[i]);
    portalPosePlaceholder[i] = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, portalPoseGlobal[i]);
  }

  // Helper lambda to add portal render node
  std::array<std::shared_ptr<PipelineTensor>, Side::Count> portalPoseLocal;
  for (size_t i = 0; i < Side::Count; ++i) {
    portalPoseLocal[i] = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline,
                                  TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    m_secureMrRenderingPipeline->assignment(portalPosePlaceholder[i], portalPoseLocal[i]);
    m_secureMrRenderingPipeline->execRenderCommand(std::make_shared<RenderCommand_Render>(portalGltfPlaceholder[i], portalPoseLocal[i], true));
  }

}

void StylizeDemo::RunPipelines() {

  // Start update loop
  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard, [this]() { return pipelineAllInitialized.load(); });
    }
    while (keepRunning.load()) {
      UpdateFrame();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

}

void StylizeDemo::UpdateFrame() {
  // OpenXR handles are externally synchronized. Controller callbacks run on
  // the render thread while this method runs on the pipeline worker, so guard
  // all calls made through the shared SecureMR framework/session.
  std::lock_guard<std::mutex> guard(secureMrApiMutex);

  auto vstRun = RunSecureMrVSTImagePipeline(nullptr);

  if (newTextureChosen)
  {
    RunSecureMrStylePredictionPipeline(nullptr);
    newTextureChosen = false;
  }

  auto styleRun = RunSecureMrStyleTransferPipeline(vstRun);
  RunSecureMrRenderingPipeline(styleRun);

}

XrSecureMrPipelineRunPICO StylizeDemo::RunSecureMrVSTImagePipeline(XrSecureMrPipelineRunPICO pre) const {
  return m_secureMrVSTImagePipeline->submit(
      {{vstOutputLeftUint8VstPlaceholder, vstOutputLeftUint8Global}},
      pre, nullptr);
}


XrSecureMrPipelineRunPICO StylizeDemo::RunSecureMrStylePredictionPipeline(XrSecureMrPipelineRunPICO pre) const {
  auto run = pre;
  for (size_t i = 0; i < Side::Count; ++i) {
    run = m_secureMrStylePredictionPipeline->submit(
        {{styleTexturePlaceholder, styleTextureGlobal[i]},
         {predictedStyleOutputPlaceholder, predictedStyleGlobal[i]}},
        run, nullptr);
  }
  return run;
}

XrSecureMrPipelineRunPICO StylizeDemo::RunSecureMrStyleTransferPipeline(XrSecureMrPipelineRunPICO pre) const {
  auto run = pre;
  for (size_t i = 0; i < Side::Count; ++i) {
    run = m_secureMrStyleTransferPipeline->submit(
        {{vstOutputLeftUint8StylePlaceholder, vstOutputLeftUint8Global},
         {predictedStyleInputPlaceholder, predictedStyleGlobal[i]},
         {stylizedImageOutputPlaceholderGPU, stylizedImageGPUGlobal[i]},
         {srcPointsPlaceholder, srcPointsGlobal[i]}},
        run, nullptr);
  }
  return run;
}

XrSecureMrPipelineRunPICO StylizeDemo::RunSecureMrRenderingPipeline(XrSecureMrPipelineRunPICO pre) const {
  std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> bindings{
      {portalGltfPlaceholder[Side::Left], portalGltf[Side::Left]},
      {portalGltfPlaceholder[Side::Right], portalGltf[Side::Right]},
      {portalPosePlaceholder[Side::Left], portalPoseGlobal[Side::Left]},
      {portalPosePlaceholder[Side::Right], portalPoseGlobal[Side::Right]}};
  return m_secureMrRenderingPipeline->submit(bindings, pre, nullptr);
}

void StylizeDemo::UpdateHandPose(const XrVector3f* leftHandDelta, const XrVector3f* rightHandDelta) {
  // TV display interaction removed
}

static std::string ToString(const XrVector3f& v) {
    return Fmt("{%.3f, %.3f, %.3f}", v.x, v.y, v.z);
}

static std::string ToString(const XrVector2f& v) {
    return Fmt("{%.3f, %.3f}", v.x, v.y);
}

static std::string ToString(const XrMatrix4x4f& m) {
    return Fmt("\n[%.3f, %.3f, %.3f, %.3f]\n[%.3f, %.3f, %.3f, %.3f]\n[%.3f, %.3f, %.3f, %.3f]\n[%.3f, %.3f, %.3f, %.3f]",
        m.m[0], m.m[4], m.m[8], m.m[12],
        m.m[1], m.m[5], m.m[9], m.m[13],
        m.m[2], m.m[6], m.m[10], m.m[14],
        m.m[3], m.m[7], m.m[11], m.m[15]);
}

static std::string ToString(const XrPosef& p) {
    return Fmt("P:{%.3f, %.3f, %.3f} Q:{%.3f, %.3f, %.3f, %.3f}", 
        p.position.x, p.position.y, p.position.z,
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
}

void StylizeDemo::UpdatePortalPose(const XrPosef& controllerPose, const XrView& view, std::shared_ptr<GlobalTensor> outputTensor, std::shared_ptr<GlobalTensor> srcPointsTensor, float xOffset) {
  XrMatrix4x4f worldFromViewCM;
  XrMatrix4x4f_CreateFromRigidTransform(&worldFromViewCM, &view.pose);

  XrMatrix4x4f viewFromWorldCM;
  XrMatrix4x4f_InvertRigidBody(&viewFromWorldCM, &worldFromViewCM);

  XrVector3f controllerWorld = controllerPose.position;
  XrVector3f controllerView;
  XrMatrix4x4f_TransformVector3f(&controllerView, &viewFromWorldCM, &controllerWorld);

  float z = controllerView.z;
  float t = (fabsf(z) > 1e-5f) ? (-2.0f / z) : 1.0f;
  XrVector3f hitView = {controllerView.x * t, controllerView.y * t, controllerView.z * t};
  
  hitView.x += xOffset;
  hitView.y += .8f;

  float hitMatrix[] = {
      1.0f, 0.0f, 0.0f, hitView.x,
      0.0f, 1.0f, 0.0f, hitView.y,
      0.0f, 0.0f, 1.0f, hitView.z,
      0.0f, 0.0f, 0.0f, 1.0f
  };

  outputTensor->setData(reinterpret_cast<int8_t*>(hitMatrix), sizeof(hitMatrix));

  if (srcPointsTensor) {
    // Calculate Affine Transform for portal
    // Portal in Model Space (assuming 1x1 quad centered at origin)
    XrVector3f cornersModel[3] = {
        {-0.51f, 0.68f, 0.0f},  // Top-Left
        {0.51f, 0.68f, 0.0f},   // Top-Right
        {-0.51f, -0.68f, 0.0f}  // Bottom-Left
    };

    XrMatrix4x4f portalPose;
    memcpy(portalPose.m, hitMatrix, sizeof(hitMatrix));

    XrMatrix4x4f modelView;
    //we need to transpose this to do our calculation using XrMatrix4x4f_TransformVector3f
    XrMatrix4x4f_Transpose(&modelView, &portalPose);

    XrMatrix4x4f projection;
    // Using GRAPHICS_OPENGL_ES as per Android/Pico standard
    XrMatrix4x4f_CreateProjectionFov(&projection, GRAPHICS_OPENGL_ES, view.fov, 0.1f, 100.0f);

    XrMatrix4x4f mvp;
    XrMatrix4x4f_Multiply(&mvp, &projection, &modelView);

    XrVector2f cornersScreen[3];
    for(int i=0; i<3; ++i) {
        XrVector3f ndc;
        XrMatrix4x4f_TransformVector3f(&ndc, &mvp, &cornersModel[i]);
        
        // Map NDC (-1..1) to VST Image Coordinates (0..2048)
        cornersScreen[i].x = (ndc.x + 1.0f) * 0.5f * (float)VST_WIDTH;
        cornersScreen[i].y = (1.0f - ndc.y) * 0.5f * (float)VST_HEIGHT;
    }

    srcPointsTensor->setData(reinterpret_cast<int8_t*>(cornersScreen), sizeof(cornersScreen));
  }
}

void StylizeDemo::UpdateControllerPose(const XrPosef* leftPose, const XrPosef* rightPose, const XrView* views, uint32_t viewCount)
{

  if (!rightPose || !views || viewCount == 0 || !globalTensorsInitialized.load()) {
    return;
  }

  std::lock_guard<std::mutex> guard(secureMrApiMutex);
  
  UpdatePortalPose(*rightPose, views[0], portalPoseGlobal[Side::Right], srcPointsGlobal[Side::Right], -0.8f);
  
  if (leftPose) {
    UpdatePortalPose(*leftPose, views[0], portalPoseGlobal[Side::Left], srcPointsGlobal[Side::Left], 0.2f);
  }
}

void StylizeDemo::HandleButtonPress(int side) {
  if (!globalTensorsInitialized.load()) {
    return;
  }

  std::lock_guard<std::mutex> guard(secureMrApiMutex);

  newTextureChosen = true;
  
  auto advanceTexture = [this](size_t index) {
    styleTextureIndex[index]++;
    if (styleTextureIndex[index] >= NUM_STYLE_TEXTURES) {
      styleTextureIndex[index] = 0;
    }
    LoadStyleTextureFromAssetPath(STYLE_TEXTURE_PATHS[styleTextureIndex[index]], styleTextureGlobal[index]);
  };

  if (side == static_cast<int>(Side::Left) || side == -1) {
    advanceTexture(Side::Left);
  }

  if (side == static_cast<int>(Side::Right) || side == -1) {
    advanceTexture(Side::Right);
  }
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<StylizeDemo>(instance, session);
}

}  // namespace SecureMR
