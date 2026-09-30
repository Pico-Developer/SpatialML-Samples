#include "mnistwild.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android_native_app_glue.h>
#include <chrono>
#include <filesystem>
#include <jni.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "securemr_utils/serialization.h"

extern AAssetManager* g_assetManager;
extern std::string g_internalDataPath;

using json = SecureMR::Json;

namespace SecureMR {
namespace {
constexpr char kInferencePipelineJson[] = "mnist_pipeline.json";
constexpr char kTensorPredictedClass[] = "predicted_class";
constexpr char kTensorPredictedScore[] = "predicted_score";
constexpr char kTensorCropImage[] = "cropped_image";

std::filesystem::path ResolveWritablePath(const std::string& fileName) {
  if (!g_internalDataPath.empty()) {
    return std::filesystem::path(g_internalDataPath) / fileName;
  }
  return {};
}

std::shared_ptr<PipelineTensor> MakeScalarTensor(const std::shared_ptr<Pipeline>& pipeline, float value) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(&value), sizeof(float));
  return tensor;
}

std::shared_ptr<PipelineTensor> MakeScalarTensor(const std::shared_ptr<Pipeline>& pipeline, uint16_t value) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(&value), sizeof(uint16_t));
  return tensor;
}

std::shared_ptr<PipelineTensor> MakeScalarTensor(const std::shared_ptr<Pipeline>& pipeline, uint8_t value) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(&value), sizeof(uint8_t));
  return tensor;
}

std::shared_ptr<PipelineTensor> MakePointTensor(const std::shared_ptr<Pipeline>& pipeline, const std::array<float, 2>& p) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 2,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(const_cast<float*>(p.data())), sizeof(float) * 2);
  return tensor;
}

std::shared_ptr<PipelineTensor> MakeColorTensor(const std::shared_ptr<Pipeline>& pipeline,
                                                const std::array<uint8_t, 8>& rgbaPair) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {2},
                                .channels = 4,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(const_cast<uint8_t*>(rgbaPair.data())), rgbaPair.size());
  return tensor;
}

std::shared_ptr<PipelineTensor> MakePoseTensor(const std::shared_ptr<Pipeline>& pipeline,
                                               const std::array<float, 16>& mat) {
  auto tensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {4, 4},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  tensor->setData(reinterpret_cast<int8_t*>(const_cast<float*>(mat.data())), sizeof(float) * mat.size());
  return tensor;
}
}  // namespace

MnistWildApp::MnistWildApp(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

MnistWildApp::~MnistWildApp() {
  keepRunning = false;
  initCv.notify_all();
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) {
      runner.join();
    }
  }
}

bool MnistWildApp::LoadAsset(const std::string& filePath, std::vector<char>& data) const {
  if (g_assetManager == nullptr) {
    Log::Write(Log::Level::Error, "LoadAsset failed: AssetManager not available");
    return false;
  }
  AAsset* asset = AAssetManager_open(g_assetManager, filePath.c_str(), AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    Log::Write(Log::Level::Error, Fmt("LoadAsset failed: unable to open %s", filePath.c_str()));
    return false;
  }
  const off_t length = AAsset_getLength(asset);
  data.resize(static_cast<size_t>(length));
  const int64_t read = AAsset_read(asset, data.data(), length);
  AAsset_close(asset);
  if (read != length) {
    Log::Write(Log::Level::Error, Fmt("LoadAsset failed: read %ld of %ld bytes", static_cast<long>(read),
                                      static_cast<long>(length)));
    data.clear();
    return false;
  }
  return true;
}

bool MnistWildApp::DeserializeInferencePipeline(const std::filesystem::path& jsonPath) {
  auto finalizeResult = [&](PipelineDeserializationResult&& result) -> bool {
    std::shared_ptr<PipelineTensor> predictedClass;
    std::shared_ptr<PipelineTensor> predictedScore;
    std::shared_ptr<PipelineTensor> croppedImage;
    try {
      predictedClass = result.tensorMap.at(kTensorPredictedClass);
      predictedScore = result.tensorMap.at(kTensorPredictedScore);
      croppedImage = result.tensorMap.at(kTensorCropImage);
    } catch (const std::exception& e) {
      Log::Write(Log::Level::Error,
                 Fmt("DeserializeInferencePipeline failed: required placeholder missing (%s)", e.what()));
      return false;
    }
    inferencePipeline = std::move(result.pipeline);
    predClassPlaceholder = std::move(predictedClass);
    predScorePlaceholder = std::move(predictedScore);
    cropImagePlaceholder = std::move(croppedImage);
    return true;
  };

  auto tryDeserialize = [&](const json& spec, const std::string& source) -> bool {
    if (!spec.is_object()) {
      Log::Write(Log::Level::Warning,
                 Fmt("DeserializeInferencePipeline failed (%s): JSON root is not an object", source.c_str()));
      return false;
    }
    PipelineDeserializationResult result;
    std::string error;
    if (!DeserializePipelineFromJson(spec, frameworkSession, result, error)) {
      Log::Write(Log::Level::Warning,
                 Fmt("DeserializeInferencePipeline failed (%s): %s", source.c_str(),
                     error.empty() ? "unknown error" : error.c_str()));
      return false;
    }
    return finalizeResult(std::move(result));
  };

  if (!jsonPath.empty()) {
    const auto pathStr = jsonPath.string();
    const json specFromFile = LoadJsonFromFile(jsonPath);
    if (tryDeserialize(specFromFile, pathStr)) {
      return true;
    }
    Log::Write(Log::Level::Info, Fmt("Falling back to asset pipeline spec after failure reading %s", pathStr.c_str()));
  }

  std::vector<char> assetBuffer;
  if (!LoadAsset(kInferencePipelineJson, assetBuffer)) {
    Log::Write(Log::Level::Error,
               Fmt("DeserializeInferencePipeline failed: unable to load asset %s", kInferencePipelineJson));
    return false;
  }

  try {
    const json assetSpec = json::parse(assetBuffer.begin(), assetBuffer.end());
    return tryDeserialize(assetSpec, Fmt("asset:%s", kInferencePipelineJson));
  } catch (const std::exception& e) {
    Log::Write(Log::Level::Error,
               Fmt("DeserializeInferencePipeline failed: cannot parse asset %s (%s)", kInferencePipelineJson,
                   e.what()));
    return false;
  }
}

void MnistWildApp::CreateFramework() {
  Log::Write(Log::Level::Info, "CreateFramework ...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, kImageWidth, kImageHeight);
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void MnistWildApp::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    CreateGlobalTensors();
    CreateInferencePipeline();
    CreateRenderPipeline();
    {
      std::lock_guard<std::mutex> lk(initMutex);
      pipelinesReady = true;
    }
    initCv.notify_all();
  });
}

void MnistWildApp::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    std::unique_lock<std::mutex> lk(initMutex);
    initCv.wait(lk, [this]() { return pipelinesReady || !keepRunning; });
    lk.unlock();
    while (keepRunning) {
      RunInferencePipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  pipelineRunners.emplace_back([this]() {
    std::unique_lock<std::mutex> lk(initMutex);
    initCv.wait(lk, [this]() { return pipelinesReady || !keepRunning; });
    lk.unlock();
    while (keepRunning) {
      RunRenderPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
  });
}

void MnistWildApp::CreateGlobalTensors() {
  Log::Write(Log::Level::Info, "Creating global tensors ...");

  predictedClassGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  int32_t defaultClass = -1;
  predictedClassGlobal->setData(reinterpret_cast<int8_t*>(&defaultClass), sizeof(defaultClass));

  predictedScoreGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float zeroScore = 0.0F;
  predictedScoreGlobal->setData(reinterpret_cast<int8_t*>(&zeroScore), sizeof(zeroScore));

  croppedImageGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {kCropHeight, kCropWidth},
                      .channels = 3,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  std::vector<uint8_t> blankImage(static_cast<size_t>(kCropHeight * kCropWidth * 3), 0);
  croppedImageGlobal->setData(reinterpret_cast<int8_t*>(blankImage.data()), blankImage.size());

  std::vector<char> gltfData;
  if (LoadAsset("tv.gltf", gltfData)) {
    gltfClassAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfScoreAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfImageAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
  } else {
    Log::Write(Log::Level::Error, "Failed to load tv.gltf");
  }

  Log::Write(Log::Level::Info, "Global tensors ready.");
}

void MnistWildApp::CreateInferencePipeline() {
  Log::Write(Log::Level::Info, "Creating inference pipeline ...");

  const std::filesystem::path jsonPath = ResolveWritablePath(kInferencePipelineJson);
  if (!DeserializeInferencePipeline(jsonPath)) {
    const std::string pathStr = jsonPath.empty() ? "<empty path>" : jsonPath.string();
    Log::Write(Log::Level::Error,
              Fmt("Failed to load inference pipeline from %s", pathStr.c_str()));
  }

  if (inferencePipeline) {
    Log::Write(Log::Level::Info, "Inference pipeline ready (mnist.tflite, target=NPU).");
  }
}

void MnistWildApp::CreateRenderPipeline() {
  Log::Write(Log::Level::Info, "Creating render pipeline ...");
  renderPipeline = std::make_shared<Pipeline>(frameworkSession);

  renderClassPlaceholder = PipelineTensor::PipelinePlaceholderLike(renderPipeline, predictedClassGlobal);
  renderScorePlaceholder = PipelineTensor::PipelinePlaceholderLike(renderPipeline, predictedScoreGlobal);
  renderCropPlaceholder = PipelineTensor::PipelinePlaceholderLike(renderPipeline, croppedImageGlobal);
  renderClassGltfPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(renderPipeline);
  renderScoreGltfPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(renderPipeline);
  renderImageGltfPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(renderPipeline);

  auto digitTextStart = MakePointTensor(renderPipeline, {0.1F, 0.3F});
  auto scoreTextStart = MakePointTensor(renderPipeline, {0.1F, 0.3F});
  auto textColors = MakeColorTensor(renderPipeline, {255, 255, 255, 255, 0, 0, 0, 255});
  auto textTextureIdClass = MakeScalarTensor(renderPipeline, static_cast<uint16_t>(0));
  auto textTextureIdScore = MakeScalarTensor(renderPipeline, static_cast<uint16_t>(0));
  auto fontSizeDigit = MakeScalarTensor(renderPipeline, 144.0F);
  auto fontSizeScore = MakeScalarTensor(renderPipeline, 144.0F);

  auto newTextureId = std::make_shared<PipelineTensor>(
      renderPipeline,
      TensorAttribute{.dimensions = {1},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});

  auto classPose = MakePoseTensor(renderPipeline,
                                  {0.5F, 0.0F, 0.0F, -0.5F,
                                   0.0F, 0.5F, 0.0F, 0.0F,
                                   0.0F, 0.0F, 0.5F, -1.5F,
                                   0.0F, 0.0F, 0.0F, 1.0F});
  auto scorePose = MakePoseTensor(renderPipeline,
                                  {0.5F, 0.0F, 0.0F, 0.5F,
                                   0.0F, 0.5F, 0.0F, 0.0F,
                                   0.0F, 0.0F, 0.5F, -1.5F,
                                   0.0F, 0.0F, 0.0F, 1.0F});
  auto imagePose = MakePoseTensor(renderPipeline,
                                  {0.5F, 0.0F, 0.0F, 0.0F,
                                   0.0F, 0.5F, 0.0F, 1.0F,
                                   0.0F, 0.0F, 0.5F, -1.5F,
                                   0.0F, 0.0F, 0.0F, 1.0F});
  auto visibleTensor = MakeScalarTensor(renderPipeline, static_cast<uint8_t>(1));

  (*renderPipeline)
      .execRenderCommand(std::make_shared<RenderCommand_DrawText>(renderClassGltfPlaceholder, "en-US",
                                                                  RenderCommand_DrawText::TypeFaceTypes::SANS_SERIF,
                                                                  1440, 960, renderClassPlaceholder, digitTextStart,
                                                                  fontSizeDigit, textColors, textTextureIdClass))
      .execRenderCommand(std::make_shared<RenderCommand_DrawText>(renderScoreGltfPlaceholder, "en-US",
                                                                  RenderCommand_DrawText::TypeFaceTypes::SANS_SERIF,
                                                                  1440, 960, renderScorePlaceholder, scoreTextStart,
                                                                  fontSizeScore, textColors, textTextureIdScore))
      .newTextureToGLTF(renderImageGltfPlaceholder, renderCropPlaceholder, newTextureId);

  auto updateMaterialCmd = std::make_shared<RenderCommand_UpdateMaterial>();
  updateMaterialCmd->gltfTensor = renderImageGltfPlaceholder;
  updateMaterialCmd->materialIds = std::vector<uint16_t>{0};
  updateMaterialCmd->attribute = RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
  updateMaterialCmd->materialValues = newTextureId;
  (*renderPipeline).execRenderCommand(updateMaterialCmd);

  auto renderClassCmd = std::make_shared<RenderCommand_Render>(renderClassGltfPlaceholder, classPose);
  renderClassCmd->visible = visibleTensor;
  (*renderPipeline).execRenderCommand(renderClassCmd);

  auto renderScoreCmd = std::make_shared<RenderCommand_Render>(renderScoreGltfPlaceholder, scorePose);
  renderScoreCmd->visible = visibleTensor;
  (*renderPipeline).execRenderCommand(renderScoreCmd);

  auto renderImageCmd = std::make_shared<RenderCommand_Render>(renderImageGltfPlaceholder, imagePose);
  renderImageCmd->visible = visibleTensor;
  (*renderPipeline).execRenderCommand(renderImageCmd);

  Log::Write(Log::Level::Info, "Render pipeline ready.");
}

void MnistWildApp::RunInferencePipeline() {
  if (!inferencePipeline) {
    return;
  }
  inferencePipeline->submit({{predClassPlaceholder, predictedClassGlobal},
                             {predScorePlaceholder, predictedScoreGlobal},
                             {cropImagePlaceholder, croppedImageGlobal}},
                            XR_NULL_HANDLE, nullptr);
}

void MnistWildApp::RunRenderPipeline() {
  if (!renderPipeline || gltfClassAsset == nullptr || gltfScoreAsset == nullptr || gltfImageAsset == nullptr) {
    return;
  }
  renderPipeline->submit({{renderClassPlaceholder, predictedClassGlobal},
                         {renderScorePlaceholder, predictedScoreGlobal},
                          {renderCropPlaceholder, croppedImageGlobal},
                          {renderClassGltfPlaceholder, gltfClassAsset},
                          {renderScoreGltfPlaceholder, gltfScoreAsset},
                          {renderImageGltfPlaceholder, gltfImageAsset}},
                         XR_NULL_HANDLE, nullptr);
}

void MnistWildApp::RequestPermission(struct android_app* app) {
  Log::Write(Log::Level::Info, "Requesting camera permission from activity");
  if (app == nullptr || app->activity == nullptr || app->activity->vm == nullptr) {
    Log::Write(Log::Level::Error, "Invalid android_app passed to RequestPermission");
    return;
  }

  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);
  if (env == nullptr) {
    Log::Write(Log::Level::Error, "Failed to attach JNI environment while requesting permission");
    return;
  }
  jobject activity = app->activity->clazz;
  jclass cls = env->GetObjectClass(activity);
  jmethodID requestCamera = env->GetMethodID(cls, "requestCameraFromNative", "()V");
  env->CallVoidMethod(activity, requestCamera);
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<MnistWildApp>(instance, session);
}

}  // namespace SecureMR
