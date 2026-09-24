// Copyright (2025) Bytedance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "whackamole_detection.h"

#include <sstream>

extern AAssetManager* g_assetManager;

namespace SecureMR {

static constexpr unsigned int NODE_COUNT = 13;

static bool LoadModelData(const std::string& filePath, std::vector<char>& modelData) {
  if (g_assetManager == nullptr) {
    Log::Write(Log::Level::Error, "LoadModelData: AssetManager is not set");
    return false;
  }

  AAsset* asset = AAssetManager_open(g_assetManager, filePath.c_str(), AASSET_MODE_BUFFER);
  if (asset == nullptr) {
    Log::Write(Log::Level::Error, Fmt("LoadModelData: Error: Failed to open asset %s", filePath.c_str()));
    return false;
  }

  off_t assetLength = AAsset_getLength(asset);
  modelData.resize(assetLength);

  AAsset_read(asset, modelData.data(), assetLength);
  AAsset_close(asset);

  return true;
}

WhackamoleDetector::WhackamoleDetector(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

WhackamoleDetector::~WhackamoleDetector() {
  keepRunning = false;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) runner.join();
  }
}

void WhackamoleDetector::CreateFramework() {
  Log::Write(Log::Level::Info, "CreateFramework ...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, 512, 512);
  startTime = std::chrono::steady_clock::now();
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void WhackamoleDetector::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    // Note: global tensors must be created before they are referred
    //       in each individual pipeline
    CreateGlobalTensor();
    CreateSecureMrVSTImagePipeline();
    CreateSecureMrModelInferencePipeline();
    CreateSecureMrRenderingPipeline();

    initialized.notify_all();
    pipelineAllInitialized = true;
  });
}

void WhackamoleDetector::UpdateHandPose(const XrVector3f* const leftHandDelta, const XrVector3f* const rightHandDelta) {
  const XrVector3f* hand = leftHandDelta;
  if (hand == nullptr) hand = rightHandDelta;
  if (hand == nullptr) return;

  CreateAndRunSecureMrMovePipeline(hand->x, hand->y, hand->z);
}

void WhackamoleDetector::CreateGlobalTensor() {
  vstOutputLeftUint8Global = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {512, 512}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  resizedLeftFp32Global = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {128, 128}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  bodyLandmarkGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {NODE_COUNT, 4, 4}, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  molesTransformsGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {MAX_MOLES, 4, 4}, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  moleCollisionsGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {MAX_MOLES, 4, 4}, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  molesVisibleGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{MAX_MOLES, XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});
  moleDeadlinesGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{MAX_MOLES, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  scoreGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  realtimeSinceStartupGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  moleOnTimingsGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{MAX_MOLES, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  moleOffTimingsGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{MAX_MOLES, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  isPoseDetectedGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});

  roiAffineGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {2, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  roiAffineUpdatedGlobal = std::make_shared<GlobalTensor>(*roiAffineGlobal);

  *roiAffineUpdatedGlobal = std::vector{0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f};

  roiImageGlobal = std::make_shared<GlobalTensor>(
          frameworkSession,
            TensorAttribute{.dimensions = {256, 256}, .channels = 3, .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO,
                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO});
  blankTextureGlobal = std::make_shared<GlobalTensor>(
          frameworkSession,
            TensorAttribute{.dimensions = {256, 256}, .channels = 4, .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO,
                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO});

  {
    std::vector<float> initMoleTransforms(MAX_MOLES * 16, 0.0f);
    for (int i = 0; i < MAX_MOLES; ++i) {
      const int base = i * 16;
      initMoleTransforms[base + 0] = 1.0f;
      initMoleTransforms[base + 5] = 1.0f;
      initMoleTransforms[base + 10] = 1.0f;
      initMoleTransforms[base + 15] = 1.0f;
      initMoleTransforms[base + 3] = -1.0f + static_cast<float>(i) * 0.5f;
      initMoleTransforms[base + 11] = -2.0f;
    }
    *molesTransformsGlobal = initMoleTransforms;
    *moleCollisionsGlobal = initMoleTransforms;
    std::vector<uint8_t> initVisible(MAX_MOLES, static_cast<uint8_t>(0));
    *molesVisibleGlobal = initVisible;
    *scoreGlobal = std::vector<int32_t>{0};
    *realtimeSinceStartupGlobal = std::vector<float>{0.0f};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distOn(2.0f, 5.0f);
    std::uniform_real_distribution<float> distOff(1.0f, 2.0f);
    std::vector<float> initOn(MAX_MOLES);
    std::vector<float> initOff(MAX_MOLES);
    std::vector<float> initDeadlines(MAX_MOLES);
    for (int i = 0; i < MAX_MOLES; ++i) {
      initOn[i] = distOn(gen);
      initOff[i] = distOff(gen);
      initDeadlines[i] = 10.0f + initOn[i];
    }
    *moleOnTimingsGlobal = initOn;
    *moleOffTimingsGlobal = initOff;
    *moleDeadlinesGlobal = initDeadlines;
  }


    std::vector<char> gltfData;
  if (LoadModelData(GLTF_PATH, gltfData)) {
    poseMarkerGltf = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    const auto initPipeline = std::make_shared<Pipeline>(frameworkSession);
    const auto initPose = std::make_shared<PipelineTensor>(
        initPipeline,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
        reinterpret_cast<int8_t*>(stagePoseData), sizeof(stagePoseData));
    const auto transitPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(initPipeline);
    initPipeline->execRenderCommand(std::make_shared<RenderCommand_Render>(transitPlaceholder, initPose, true));
    initPipeline->submit({{transitPlaceholder, poseMarkerGltf}}, XR_NULL_HANDLE, nullptr);

  } else {
    Log::Write(Log::Level::Error, "Failed to load glTF data from file.");
  }

  std::vector<char> scoreboardGltfData;
  if (LoadModelData(GLTF_SCOREBOARD_PATH, scoreboardGltfData)) {
    scoreboardGltf = std::make_shared<GlobalTensor>(frameworkSession, scoreboardGltfData.data(), scoreboardGltfData.size());
    const auto initPipeline2 = std::make_shared<Pipeline>(frameworkSession);
    const auto scoreboardPose = std::make_shared<PipelineTensor>(
        initPipeline2,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    float scoreboardPoseData[]{0.8f, 0.0f, 0.0f, -0.3f, 0.0f, 0.8f, 0.0f, 0.2f, 0.0f, 0.0f, 0.8f, -1.4f, 0.0f, 0.0f, 0.0f, 1.0f};
    scoreboardPose->setData(reinterpret_cast<int8_t*>(scoreboardPoseData), sizeof(scoreboardPoseData));
    const auto scoreboardPlaceholder = PipelineTensor::PipelineGLTFPlaceholder(initPipeline2);

    const auto newTextureId = std::make_shared<PipelineTensor>(
            initPipeline2,
            TensorAttribute{.dimensions = {1}, .channels = 1, .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
    const auto blankTexturePh = PipelineTensor::PipelinePlaceholderLike(initPipeline2, blankTextureGlobal);

    initPipeline2->execRenderCommand(std::make_shared<RenderCommand_Render>(scoreboardPlaceholder, scoreboardPose, true));
    initPipeline2->newTextureToGLTF(scoreboardPlaceholder, blankTexturePh, newTextureId);

    auto updateMaterialCmd = std::make_shared<RenderCommand_UpdateMaterial>();
    updateMaterialCmd->gltfTensor = scoreboardPlaceholder;
    updateMaterialCmd->materialIds = std::vector<uint16_t>{0};
    updateMaterialCmd->attribute = RenderCommand_UpdateMaterial::MaterialAttribute::TEXTURE_BASE_COLOR;
    updateMaterialCmd->materialValues = newTextureId;
    (*initPipeline2).execRenderCommand(updateMaterialCmd);

    initPipeline2->submit({{scoreboardPlaceholder, scoreboardGltf}, {blankTexturePh, blankTextureGlobal}}, XR_NULL_HANDLE, nullptr);
  } else {
    Log::Write(Log::Level::Error, "Failed to load scoreboard glTF data from file.");
  }

  std::vector<char> duckGltfData;
  if (LoadModelData(GLTF_DUCK_PATH, duckGltfData)) {
    for (int i = 0; i < MAX_MOLES; ++i) {
      duckGltfs[i] = std::make_shared<GlobalTensor>(frameworkSession, duckGltfData.data(), duckGltfData.size());
    }
  } else {
    Log::Write(Log::Level::Error, "Failed to load duck glTF data from file.");
  }

  std::vector<char> redboxGltfData;
  if (LoadModelData(GLTF_REDBOX_PATH, redboxGltfData)) {
      for (int i = 0; i < MAX_JOINTS; ++i) {
          redboxGltfs[i] = std::make_shared<GlobalTensor>(frameworkSession, redboxGltfData.data(), redboxGltfData.size());
      }
  } else {
      Log::Write(Log::Level::Error, "Failed to load redbox glTF data from file.");
  }

}

void WhackamoleDetector::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrVSTImagePipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrModelInferencePipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrRenderingPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
}

void WhackamoleDetector::CreateSecureMrVSTImagePipeline() {
  Log::Write(Log::Level::Info, "SpatialML CreateSecureMrVSTImagePipeline");

  m_secureMrVSTImagePipeline = std::make_shared<Pipeline>(frameworkSession);

  vstOutputLeftUint8Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftUint8Global);
  vstOutputLeftFp32Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, resizedLeftFp32Global);

  static float RESHAPE_MAT_512_TO_128[]{0.25f, 0.0f, 0.0f, 0.0f, 0.25f, 0.0f};
  const auto affineMatReshape = std::make_shared<PipelineTensor>(
      m_secureMrVSTImagePipeline,
      TensorAttribute{.dimensions = {2, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(RESHAPE_MAT_512_TO_128), sizeof(RESHAPE_MAT_512_TO_128));

  const auto resizedLeftUint8 = std::make_shared<PipelineTensor>(
      m_secureMrVSTImagePipeline,
      TensorAttribute{.dimensions = {128, 128}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});

  (*m_secureMrVSTImagePipeline)
      .cameraAccess(nullptr, vstOutputLeftUint8Placeholder, nullptr, nullptr)
      .applyAffine(affineMatReshape, vstOutputLeftUint8Placeholder, resizedLeftUint8)
      .assignment(resizedLeftUint8, vstOutputLeftFp32Placeholder)
      .arithmetic("({0} / 255.0)", {vstOutputLeftFp32Placeholder}, vstOutputLeftFp32Placeholder);
}

void WhackamoleDetector::CreateSecureMrModelInferencePipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrModelInferencePipeline");

  m_secureMrDetectionPipeline = std::make_shared<Pipeline>(frameworkSession);
  m_secureMrLandmarkPipeline = std::make_shared<Pipeline>(frameworkSession);
  m_secureMrAffineUpdatePipeline = std::make_shared<Pipeline>(frameworkSession);

  // Step 1: pipeline placeholders for global tensors
  smallF32ImagePlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrDetectionPipeline, resizedLeftFp32Global);
  largeU8ImagePlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, vstOutputLeftUint8Global);
  isPoseDetectedPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrDetectionPipeline, isPoseDetectedGlobal);
  bodyLandmarkPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, bodyLandmarkGlobal);
  moleCollisionsPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, moleCollisionsGlobal);
  molesVisiblePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, molesVisibleGlobal);
  moleDeadlinesPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, moleDeadlinesGlobal);
  scorePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, scoreGlobal);
  realtimeSinceStartupPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, realtimeSinceStartupGlobal);
  moleOnTimingsPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, moleOnTimingsGlobal);
  moleOffTimingsPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, moleOffTimingsGlobal);

  roiAffinePh1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrDetectionPipeline, roiAffineGlobal);
  roiAffinePh2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrAffineUpdatePipeline, roiAffineGlobal);
  roiAffinePh3 = PipelineTensor::PipelinePlaceholderLike(m_secureMrAffineUpdatePipeline, roiAffineUpdatedGlobal);
  roiAffinePh4 = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, roiAffineUpdatedGlobal);

  roiImagePh = PipelineTensor::PipelinePlaceholderLike(m_secureMrLandmarkPipeline, roiImageGlobal);

  // Step 2: local tensors
  auto poseAnchor = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {896, 12}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto poseScores = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute_ScalarArray{.size = 896, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto bestPoseScore = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto anchorMatTensor = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {896, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  /**
   * poseAnchors[:, 4:8] -> center X, center Y, head X, head Y to compute the radius of the ROI, using the
   * Vitruvian man model
   */
  auto poseKeypointAll = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {896, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto bestPoseIndex = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {1, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto bestPoseIndexPlusOne = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {1, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto bestPoseSrcSlice2 =
      std::make_shared<PipelineTensor>(m_secureMrDetectionPipeline, TensorAttribute_SliceArray{.size = 2});
  auto bestPoseSrcSlice1 =
      std::make_shared<PipelineTensor>(m_secureMrDetectionPipeline, TensorAttribute_SliceArray{.size = 1});
  const auto bestKeypointFloat = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline, TensorAttribute_Point2Array{2, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  const auto bestHipKeypoint = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {1, 2}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  const auto bestHeadKeypoint = std::make_shared<PipelineTensor>(*bestHipKeypoint);
  const auto bestKeypointVec = std::make_shared<PipelineTensor>(*bestHipKeypoint);
  const auto bestKeypointVecPerp = std::make_shared<PipelineTensor>(*bestHipKeypoint);
  const auto bestKeypointVecMultiplier = std::make_shared<PipelineTensor>(*bestHipKeypoint);
  const auto bestLeftKeypoint = std::make_shared<PipelineTensor>(*bestHipKeypoint);
  const auto roiPoints = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute_Point2Array{.size = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  const auto affineMatReshape = std::make_shared<PipelineTensor>(
      m_secureMrDetectionPipeline,
      TensorAttribute{.dimensions = {2, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  const auto roiImage = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {256, 256}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  const auto roiImageFp32 = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {256, 256}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  const auto skeletonLandmarks = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {39, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  const auto landmarkTVec = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {1, 3}, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  std::array<std::shared_ptr<PipelineTensor>, NODE_COUNT> landmarkTVecLast{};
  for (auto& each : landmarkTVecLast) {
    static float DEFAULT_XYZ[]{0.0f, 0.0f, 0.0f};
    each = std::make_shared<PipelineTensor>(
        m_secureMrLandmarkPipeline,
        TensorAttribute{.dimensions = {1, 3}, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
        reinterpret_cast<int8_t*>(DEFAULT_XYZ), sizeof(DEFAULT_XYZ));
  }
  const auto landmarkMat = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {4, 4}, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  static float DEFAULT_R_VEC[]{0.0, 0.0, 0.0};
  const auto rvecTensor = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {1, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_R_VEC), sizeof(DEFAULT_R_VEC));
  static float DEFAULT_XYZ_MULTIPLIER[]{1.0, -1.0, -1.0};
  const auto yReverse = std::make_shared<PipelineTensor>(
      m_secureMrLandmarkPipeline,
      TensorAttribute{.dimensions = {1, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_XYZ_MULTIPLIER), sizeof(DEFAULT_XYZ_MULTIPLIER));

  // Step 2(+): Set init data to local tensors
  static float RESHAPE_MAT_128_TO_512[]{4.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f};
  *bestPoseSrcSlice1 = std::vector<int>{0, -1};         // init data for bestPoseSrcSlice: [0:-1]
  *bestPoseSrcSlice2 = std::vector<int>{0, -1, 0, -1};  // init data for bestPoseSrcSlice: [0:-1, 0:-1]
  affineMatReshape->setData(reinterpret_cast<int8_t*>(RESHAPE_MAT_128_TO_512), sizeof(RESHAPE_MAT_128_TO_512));
  *bestKeypointVecMultiplier = std::vector{1.0f, -1.0f};
  if (std::vector<char> anchorData; LoadModelData(ANCHOR_MAT, anchorData)) {
    anchorMatTensor->setData(reinterpret_cast<int8_t*>(anchorData.data()), anchorData.size());
  } else {
    Log::Write(Log::Level::Error, "Failed to load anchor.mat data from file.");
  }

  // Step 3: Assembly!
  if (std::vector<char> modelData1, modelData2;
      LoadModelData(WHACKAMOLE_DETECTION_MODEL_PATH, modelData1) && LoadModelData(WHACKAMOLE_LANDMARK_MODEL_PATH, modelData2)) {
    (*m_secureMrDetectionPipeline)
        .runAlgorithm(modelData1.data(), modelData1.size(), {{"image", smallF32ImagePlaceholder}}, {},
                      {{"pose_anchor", poseAnchor}, {"score", poseScores}},
                      {{"pose_anchor", "box_coords"}, {"score", "box_scores"}}, "pose")
        .assignment((*poseAnchor)[{{0, -1}, {4, 8}}], poseKeypointAll)
        .arithmetic("({0} / 128.0 + {1}) * 128.0", {poseKeypointAll, anchorMatTensor}, poseKeypointAll)
        .argMax(poseScores, bestPoseIndex)
        .arithmetic("({0} + 1)", {bestPoseIndex}, bestPoseIndexPlusOne)
        .assignment(bestPoseIndex, (*bestPoseSrcSlice2)[0][0])
        .assignment(bestPoseIndexPlusOne, (*bestPoseSrcSlice2)[0][1])
        .assignment((*bestPoseSrcSlice2)[0], bestPoseSrcSlice1)
        .assignment((*poseKeypointAll)[bestPoseSrcSlice2], bestKeypointFloat)
        .applyAffinePoint(affineMatReshape, bestKeypointFloat, bestKeypointFloat)
        .assignment((*poseScores)[bestPoseSrcSlice1], bestPoseScore)
        .compareTo(*bestPoseScore > std::vector<float>{0.0}, isPoseDetectedPlaceholder)
        .assignment((*bestKeypointFloat)[0], bestHipKeypoint)
        .assignment((*bestKeypointFloat)[1], bestHeadKeypoint)
        .arithmetic("{0} - {1}", {bestHeadKeypoint, bestHipKeypoint}, bestKeypointVec)
        .elementwise(Pipeline::ElementwiseOp::MULTIPLY, {bestKeypointVec, bestKeypointVecMultiplier}, bestKeypointVec)
        .assignment((*bestKeypointVec)[std::vector{0, 0}], (*bestKeypointVecPerp)[std::vector{0, 1}])
        .assignment((*bestKeypointVec)[std::vector{0, 1}], (*bestKeypointVecPerp)[std::vector{0, 0}])
        .arithmetic("{0} + {1}", {bestHipKeypoint, bestKeypointVecPerp}, bestLeftKeypoint)
        .assignment(bestHipKeypoint, (*roiPoints)[0])
        .assignment(bestHeadKeypoint, (*roiPoints)[1])
        .assignment(bestLeftKeypoint, (*roiPoints)[2])
        .getAffine(roiPoints, std::array{128.0f, 128.0f, 128.0f, 0.0f, 255.0f, 128.0f}, roiAffinePh1);

    m_secureMrAffineUpdatePipeline->assignment(roiAffinePh2, roiAffinePh3);

    (*m_secureMrLandmarkPipeline)
        .applyAffine(roiAffinePh4, largeU8ImagePlaceholder, roiImagePh)
        .assignment(roiImagePh, roiImageFp32)
        .arithmetic("({0} - 127.5)/ 127.5", {roiImageFp32}, roiImageFp32)
        .runAlgorithm(modelData2.data(), modelData2.size(), {{"input_1", roiImageFp32}}, {},
                      {{"landmarks", skeletonLandmarks}}, {{"landmarks", "Identity_4"}}, "pose_landmark");



    // Add javascript operator
      const auto stagePoseTensor = std::make_shared<PipelineTensor>(
              m_secureMrLandmarkPipeline,
              TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
              reinterpret_cast<int8_t*>(stagePoseData), sizeof(stagePoseData));


      if (std::vector<char> javascriptCode;
          LoadModelData(JOINT_TRANSFORMER_JSCODE_PATH, javascriptCode))
    {
      m_secureMrLandmarkPipeline->runJavascript(javascriptCode.data(), javascriptCode.size(),
                                                {{"positionArray", skeletonLandmarks},
                                                 {"moleCollisions", moleCollisionsPlaceholder},
                                                 {"realtimesincestartup", realtimeSinceStartupPlaceholder},
                                                 {"moleOnTimings", moleOnTimingsPlaceholder},
                                                 {"moleOffTimings", moleOffTimingsPlaceholder},
                                                 {"moleVisibleArray", molesVisiblePlaceholder},
                                                 {"moleDeadlines", moleDeadlinesPlaceholder},
                                                 {"score", scorePlaceholder},
                                                 {"stagePose", stagePoseTensor}},
                                                {{"matrixArray", bodyLandmarkPlaceholder},
                                                 {"moleVisibleArray", molesVisiblePlaceholder},
                                                 {"moleDeadlines", moleDeadlinesPlaceholder},
                                                 {"score", scorePlaceholder}}
      );

    }

        
  } else {
    Log::Write(Log::Level::Error, "Failed to load model data from file.");
  }
}

void WhackamoleDetector::CreateSecureMrRenderingPipeline() {
  m_secureMrRenderingPipeline = std::make_shared<Pipeline>(frameworkSession);

  // Step 1: placeholders
  isPoseDetectedPlaceholder2 =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, isPoseDetectedGlobal);
  gltfPlaceholderTensor = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, poseMarkerGltf);
  bodyLandmarkPlaceholder2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, bodyLandmarkGlobal);
  gltfScoreboardPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, scoreboardGltf);
  molesTransformsPlaceholder2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, molesTransformsGlobal);
  molesVisiblePlaceholder2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, molesVisibleGlobal);
  currentScorePh = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, scoreGlobal);

  // Step 2 : Assembly!
  m_secureMrRenderingPipeline->execRenderCommand(std::make_shared<RenderCommand_UpdateNodesLocalPoses>(
      gltfPlaceholderTensor, std::vector<uint16_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      bodyLandmarkPlaceholder2));

  for (int i = 0; i < MAX_MOLES; ++i) {
    moleGltfPlaceholders[i] = PipelineTensor::PipelineGLTFPlaceholder(m_secureMrRenderingPipeline);
    molePoseTensors[i] = std::make_shared<PipelineTensor>(
        m_secureMrRenderingPipeline,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    moleVisibleTensors[i] = std::make_shared<PipelineTensor>(
        m_secureMrRenderingPipeline,
        TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});
    (*m_secureMrRenderingPipeline)
        .assignment((*molesVisiblePlaceholder2)[std::vector{static_cast<int>(i)}], moleVisibleTensors[i]);
    (*m_secureMrRenderingPipeline)
        .assignment((*molesTransformsPlaceholder2)[{{i, i + 1}, {0, 4}, {0, 4}}], molePoseTensors[i]);

    auto renderCmd = std::make_shared<RenderCommand_Render>(moleGltfPlaceholders[i], molePoseTensors[i], true);
    renderCmd->visible = moleVisibleTensors[i];
    m_secureMrRenderingPipeline->execRenderCommand(renderCmd);
  }


  const auto stagePoseTensor = std::make_shared<PipelineTensor>(
            m_secureMrRenderingPipeline,
            TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
            reinterpret_cast<int8_t*>(stagePoseData), sizeof(stagePoseData));

   static int whackerindices[4]{2,3,8,9};

    for (int i = 0; i < MAX_JOINTS; ++i) {
        redboxGltfPlaceholders[i] = PipelineTensor::PipelineGLTFPlaceholder(m_secureMrRenderingPipeline);
        redboxPoseTensors[i] = std::make_shared<PipelineTensor>(
                m_secureMrRenderingPipeline,
                TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

         const int w = whackerindices[i]; 
        (*m_secureMrRenderingPipeline)
                .assignment((*bodyLandmarkPlaceholder2)[{{w, w + 1}, {0, 4}, {0, 4}}], redboxPoseTensors[i]);
        (*m_secureMrRenderingPipeline)
                .arithmetic("{0} * {1}",{stagePoseTensor, redboxPoseTensors[i]}, redboxPoseTensors[i]);



        auto renderCmd = std::make_shared<RenderCommand_Render>(redboxGltfPlaceholders[i], redboxPoseTensors[i], true);
        m_secureMrRenderingPipeline->execRenderCommand(renderCmd);
    }


  (*m_secureMrRenderingPipeline)
      .debugRenderText(gltfScoreboardPlaceholder, currentScorePh, 256, 256, std::tuple<float, float>{0.5F, 0.5F}, 64.0F,
                       std::array<std::array<uint8_t, 4>, 2>{{{255, 255, 255, 255}, {0, 0, 0, 255}}},
                       static_cast<uint16_t>(1));
}

XrSecureMrPipelineRunPICO WhackamoleDetector::RunSecureMrVSTImagePipeline(const XrSecureMrPipelineRunPICO pre) {
  return m_secureMrVSTImagePipeline->submit({{vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global},
                                             {vstOutputLeftFp32Placeholder, resizedLeftFp32Global}},
                                            pre, nullptr);
}

XrSecureMrPipelineRunPICO WhackamoleDetector::RunSecureMrModelInferencePipeline(const XrSecureMrPipelineRunPICO pre) {
  const auto detection = m_secureMrDetectionPipeline->submit({{smallF32ImagePlaceholder, resizedLeftFp32Global},
                                                              {isPoseDetectedPlaceholder, isPoseDetectedGlobal},
                                                              {roiAffinePh1, roiAffineGlobal}},
                                                             pre, nullptr);
  const auto affine = m_secureMrAffineUpdatePipeline->submit(
      {{roiAffinePh2, roiAffineGlobal}, {roiAffinePh3, roiAffineUpdatedGlobal}}, detection, isPoseDetectedGlobal);
  const auto now = std::chrono::steady_clock::now();
  const float elapsed = std::chrono::duration<float>(now - startTime).count();
  *realtimeSinceStartupGlobal = std::vector<float>{elapsed};
  if (elapsed >= 10.0f && !molesInitApplied) {
    CreateAndRunInitMolesPipeline();
    molesInitApplied = true;
  }
  {
    std::vector<float> onTimings(MAX_MOLES);
    std::vector<float> offTimings(MAX_MOLES);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distOn(2.0f, 5.0f);
    std::uniform_real_distribution<float> distOff(300.0f, 312.0f);
    for (int i = 0; i < MAX_MOLES; ++i) {
      onTimings[i] = distOn(gen);
      offTimings[i] = distOff(gen);
    }
    *moleOnTimingsGlobal = onTimings;
    *moleOffTimingsGlobal = offTimings;
  }
  return m_secureMrLandmarkPipeline->submit({{largeU8ImagePlaceholder, vstOutputLeftUint8Global},
                                             {bodyLandmarkPlaceholder, bodyLandmarkGlobal},
                                             {moleCollisionsPlaceholder, moleCollisionsGlobal},
                                             {molesVisiblePlaceholder, molesVisibleGlobal},
                                             {moleDeadlinesPlaceholder, moleDeadlinesGlobal},
                                             {realtimeSinceStartupPlaceholder, realtimeSinceStartupGlobal},
                                             {moleOnTimingsPlaceholder, moleOnTimingsGlobal},
                                             {moleOffTimingsPlaceholder, moleOffTimingsGlobal},
                                             {scorePlaceholder, scoreGlobal},
                                             {roiAffinePh4, roiAffineUpdatedGlobal},
                                             {roiImagePh, roiImageGlobal}},
                                            affine, nullptr);
}

void WhackamoleDetector::CreateAndRunInitMolesPipeline() {
  if (m_secureMrInitMolesPipeline == nullptr) {
    m_secureMrInitMolesPipeline = std::make_shared<Pipeline>(frameworkSession);
    bodyLandmarkInitPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrInitMolesPipeline, bodyLandmarkGlobal);
    molesTransformsInitPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrInitMolesPipeline, molesTransformsGlobal);
    molesCollisionsInitPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrInitMolesPipeline, moleCollisionsGlobal);
  }

  const auto mat11 = std::make_shared<PipelineTensor>(
      m_secureMrInitMolesPipeline,
      TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  const auto mat12 = std::make_shared<PipelineTensor>(*mat11);
  const auto centerAvgMat = std::make_shared<PipelineTensor>(*mat11);
  const auto worldMat = std::make_shared<PipelineTensor>(*mat11);
  const auto stagePoseInit = std::make_shared<PipelineTensor>(
      m_secureMrInitMolesPipeline,
      TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(stagePoseData), sizeof(stagePoseData));

  

    (*m_secureMrInitMolesPipeline)
       .assignment(stagePoseInit, worldMat);

  for (int i = 0; i < MAX_MOLES; ++i) {
    const auto moleMat = std::make_shared<PipelineTensor>(
        m_secureMrInitMolesPipeline,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

    float ox = 0.0f, oy = 0.0f, oz = 0.0f;
    switch (i) {
      case 0: ox = -0.4f; oy = 0.5f; oz = 0.0f; break;
      case 1: ox = 0.4f;  oy = 0.5f; oz = 0.0f; break;
      case 2: ox = -0.4f; oy = 0.0f; oz = 0.0f; break;
      case 3: ox = 0.4f;  oy = 0.0f; oz = 0.0f; break;
      case 4: ox = 0.0f;  oy = 0.9f; oz = 0.0f; break;
      default: ox = 0.0f; oy = 0.0f; oz = 0.0f; break;
    }
    float offsetData[16]{
        1.0f, 0.0f, 0.0f, ox,
        0.0f, 1.0f, 0.0f, oy,
        0.0f, 0.0f, 1.0f, oz,
        0.0f, 0.0f, 0.0f, 1.0f};
    const auto offsetMat = std::make_shared<PipelineTensor>(
        m_secureMrInitMolesPipeline,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
        reinterpret_cast<int8_t*>(offsetData), sizeof(offsetData));
    const auto collisionMat = std::make_shared<PipelineTensor>(
        m_secureMrInitMolesPipeline,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

    (*m_secureMrInitMolesPipeline)
        .arithmetic("{0} * {1}", {offsetMat, worldMat}, moleMat)
        .assignment(offsetMat, collisionMat)
        .assignment(moleMat, (*molesTransformsInitPlaceholder)[{{i, i + 1}, {0, 4}, {0, 4}}])
        .assignment(collisionMat, (*molesCollisionsInitPlaceholder)[{{i, i + 1}, {0, 4}, {0, 4}}]);
  }

  m_secureMrInitMolesPipeline->submit({{bodyLandmarkInitPlaceholder, bodyLandmarkGlobal},
                                       {molesTransformsInitPlaceholder, molesTransformsGlobal},
                                       {molesCollisionsInitPlaceholder, moleCollisionsGlobal}},
                                      XR_NULL_HANDLE, nullptr);
}

XrSecureMrPipelineRunPICO WhackamoleDetector::RunSecureMrRenderingPipeline(const XrSecureMrPipelineRunPICO pre) {
  return m_secureMrRenderingPipeline->submit({{gltfPlaceholderTensor, poseMarkerGltf},
                                              {isPoseDetectedPlaceholder2, isPoseDetectedGlobal},
                                              {bodyLandmarkPlaceholder2, bodyLandmarkGlobal},
                                              {gltfScoreboardPlaceholder, scoreboardGltf},
                                              {molesTransformsPlaceholder2, molesTransformsGlobal},
                                              {molesVisiblePlaceholder2, molesVisibleGlobal},
                                              {moleGltfPlaceholders[0], duckGltfs[0]},
                                              {moleGltfPlaceholders[1], duckGltfs[1]},
                                              {moleGltfPlaceholders[2], duckGltfs[2]},
                                              {moleGltfPlaceholders[3], duckGltfs[3]},
                                              {moleGltfPlaceholders[4], duckGltfs[4]},
                                              {redboxGltfPlaceholders[0], redboxGltfs[0]},
                                              {redboxGltfPlaceholders[1], redboxGltfs[1]},
                                              {redboxGltfPlaceholders[2], redboxGltfs[2]},
                                              {redboxGltfPlaceholders[3], redboxGltfs[3]},
                                              {currentScorePh, scoreGlobal}},
                                             pre, nullptr);
}

void WhackamoleDetector::CreateAndRunSecureMrMovePipeline(const float x, const float y, const float z) {
  std::ostringstream oss;
  oss << "updated hand-pose delta {" << x << ',' << y << ',' << z << '}';
  Log::Write(Log::Level::Info, oss.str());
  if (!pipelineAllInitialized) return;
  if (m_secureMrMovePipeline == nullptr) {
    m_secureMrMovePipeline = std::make_shared<Pipeline>(frameworkSession);
    gltfPlaceholderTensor2 = PipelineTensor::PipelineGLTFPlaceholder(m_secureMrMovePipeline);

    stagePose = std::make_shared<PipelineTensor>(
        m_secureMrMovePipeline,
        TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

    m_secureMrMovePipeline->execRenderCommand(
        std::make_shared<RenderCommand_UpdatePose>(gltfPlaceholderTensor2, stagePose));
  }

  stagePoseData[3] += x;
  stagePoseData[7] += y;
  stagePoseData[11] += z;
  stagePose->setData(reinterpret_cast<int8_t*>(stagePoseData), sizeof(stagePoseData));
  m_secureMrMovePipeline->submit({{gltfPlaceholderTensor2, poseMarkerGltf}}, nullptr, nullptr);
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<WhackamoleDetector>(instance, session);
}
}  // namespace SecureMR
