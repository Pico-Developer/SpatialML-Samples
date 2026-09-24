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

#include "face_tracking.h"

extern AAssetManager* g_assetManager;

namespace SecureMR {

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

FaceTracker::FaceTracker(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

FaceTracker::~FaceTracker() {
  keepRunning = false;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) runner.join();
  }
}

void FaceTracker::CreateFramework() {
  Log::Write(Log::Level::Info, "CreateFramework ...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, 256, 256);
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void FaceTracker::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    // Note: global tensors must be created before they are referred
    //       in each individual pipeline
    CreateGlobalTensor();
    CreateSecureMrVSTImagePipeline();
    CreateSecureMrModelInferencePipeline();
    CreateSecureMrMap2dTo3dPipeline();
    CreateSecureMrRenderingPipeline();

    initialized.notify_all();
    pipelineAllInitialized = true;
  });
}

void FaceTracker::CreateGlobalTensor() {
  vstOutputLeftUint8Global = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {256, 256}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  vstOutputRightUint8Global = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {256, 256}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  vstOutputLeftFp32Global = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {256, 256}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  vstTimestampGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute_TimeStamp{});
  vstCameraMatrixGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {3, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  uvGlobal = std::make_shared<GlobalTensor>(frameworkSession,
                                            TensorAttribute_Point2Array{1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  isFaceDetectedGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute_ScalarArray{1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});

  static float DEFAULT_EYE_4x4_MAT[]{
      1.0f, 0.0f, 0.0f, 0.0f,  //
      0.0f, 1.0f, 0.0f, 0.0f,  //
      0.0f, 0.0f, 1.0f, 0.0f,  //
      0.0f, 0.0f, 0.0f, 1.0f   //
  };
  currentPositionGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_EYE_4x4_MAT), sizeof(DEFAULT_EYE_4x4_MAT));
  previousPositionGlobal = std::make_shared<GlobalTensor>(
      frameworkSession,
      TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_EYE_4x4_MAT), sizeof(DEFAULT_EYE_4x4_MAT));

  std::vector<char> gltfData;
  if (LoadModelData(GLTF_PATH, gltfData)) {
    gltfAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
  } else {
    Log::Write(Log::Level::Error, "Failed to load glTF data from file.");
  }
}

void FaceTracker::RunPipelines() {
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
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrMap2dTo3dPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

void FaceTracker::CreateSecureMrVSTImagePipeline() {
  Log::Write(Log::Level::Info, "SpatialML CreateSecureMrVSTImagePipeline");

  m_secureMrVSTImagePipeline = std::make_shared<Pipeline>(frameworkSession);

  vstOutputLeftUint8Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftUint8Global);
  vstOutputRightUint8Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputRightUint8Global);
  vstTimestampPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstTimestampGlobal);
  vstCameraMatrixPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstCameraMatrixGlobal);

  vstOutputLeftFp32Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftFp32Global);

  (*m_secureMrVSTImagePipeline)
      .cameraAccess(vstOutputLeftUint8Placeholder, vstOutputRightUint8Placeholder, vstTimestampPlaceholder,
                    vstCameraMatrixPlaceholder)
      .assignment(vstOutputLeftUint8Placeholder, vstOutputLeftFp32Placeholder)
      .arithmetic("({0} / 255.0)", {vstOutputLeftFp32Placeholder}, vstOutputLeftFp32Placeholder);
}

void FaceTracker::CreateSecureMrModelInferencePipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrModelInferencePipeline");

  m_secureMrModelInferencePipeline = std::make_shared<Pipeline>(frameworkSession);

  // Step 1: pipeline placeholders for global tensors
  vstImagePlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, vstOutputLeftFp32Global);
  uvPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, uvGlobal);
  isFaceDetectedPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, isFaceDetectedGlobal);

  // Step 2: local tensors
  auto faceAnchor = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute{.dimensions = {896, 16}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto faceScores = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute_ScalarArray{.size = 896, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto bestFaceScore = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto anchorMatTensor = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute{.dimensions = {896, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto faceLandmarks = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute{.dimensions = {896, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto bestFaceIndex = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute{.dimensions = {1, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto bestFaceIndexPlusOne = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute{.dimensions = {1, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto bestFaceSrcSlice2 =
      std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute_SliceArray{.size = 2});
  auto bestFaceSrcSlice1 =
      std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute_SliceArray{.size = 1});

  static int DEFAULT_UV_THRESHOLD[] = {20, 20};
  auto uvThreshold = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute_Point2Array{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  uvThreshold->setData(reinterpret_cast<int8_t*>(DEFAULT_UV_THRESHOLD), sizeof(DEFAULT_UV_THRESHOLD));
  auto uvDetected = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute_Point2Array{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto uvDetectedAll = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto scoreDetected = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute_ScalarArray{.size = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto temp = std::make_shared<PipelineTensor>(
      m_secureMrModelInferencePipeline,
      TensorAttribute{.dimensions = {2, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});

  // Step 2(+): Set init data to local tensors
  *bestFaceSrcSlice1 = std::vector<int>{0, -1};        // init data for bestFaceSrcSlice: [0:-1]
  *bestFaceSrcSlice2 = std::vector<int>{0, -1, 0, 2};  // init data for bestFaceSrcSlice: [0:-1, 0:2]
  if (std::vector<char> anchorData; LoadModelData(ANCHOR_MAT, anchorData)) {
    anchorMatTensor->setData(reinterpret_cast<int8_t*>(anchorData.data()), anchorData.size());
  } else {
    Log::Write(Log::Level::Error, "Failed to load anchor.mat data from file.");
  }

  // Step 3: Assembly!
  if (std::vector<char> modelData; LoadModelData(FACE_DETECTION_MODEL_PATH, modelData)) {
    (*m_secureMrModelInferencePipeline)
        .runAlgorithm(modelData.data(), modelData.size(), {{"image", vstImagePlaceholder}}, {},
                      {{"face_anchor", faceAnchor}, {"score", faceScores}},
                      {{"face_anchor", "box_coords"}, {"score", "box_scores"}}, "face")
        .assignment((*anchorMatTensor)[{{0, -1}, {0, 2}}], (*anchorMatTensor)[{{0, -1}, {2, 4}}])
        .assignment((*faceAnchor)[{{0, -1}, {4, 8}}], faceLandmarks)
        .arithmetic("({0} / 256.0 + {1}) * 256.0", {faceLandmarks, anchorMatTensor}, faceLandmarks)
        .argMax(faceScores, bestFaceIndex)
        .arithmetic("({0} + 1)", {bestFaceIndex}, bestFaceIndexPlusOne)
        .assignment(bestFaceIndex, (*bestFaceSrcSlice2)[0][0])
        .assignment(bestFaceIndexPlusOne, (*bestFaceSrcSlice2)[0][1])
        .assignment((*bestFaceSrcSlice2)[0], bestFaceSrcSlice1)
        .assignment((*faceLandmarks)[bestFaceSrcSlice2], uvPlaceholder)
        .assignment((*faceScores)[bestFaceSrcSlice1], bestFaceScore)
        .compareTo(*bestFaceScore > std::vector<float>{0.55}, isFaceDetectedPlaceholder)
        .compareTo(*uvPlaceholder > uvThreshold, uvDetected)
        .compareTo(*bestFaceScore > std::vector<float>{0.55}, scoreDetected)
        .all(uvDetected, uvDetectedAll)
        .assignment(uvDetectedAll, (*temp)[{{0, 1}, {0, 1}}])
        .assignment(scoreDetected, (*temp)[{{1, 2}, {0, 1}}])
        .all(temp, isFaceDetectedPlaceholder);

  } else {
    Log::Write(Log::Level::Error, "Failed to load model data from file.");
  }
}

void FaceTracker::CreateSecureMrMap2dTo3dPipeline() {
  m_secureMrMap2dTo3dPipeline = std::make_shared<Pipeline>(frameworkSession);

  // Step 1: pipeline placeholders
  uvPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, uvGlobal);
  timestampPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstTimestampGlobal);
  cameraMatrixPlaceholder1 =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstCameraMatrixGlobal);
  leftImgePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstOutputLeftUint8Global);
  rightImagePlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstOutputRightUint8Global);
  currentPositionPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, currentPositionGlobal);

  // Step 2: local tensors
  auto pointXYZ = std::make_shared<PipelineTensor>(
      m_secureMrMap2dTo3dPipeline,
      TensorAttribute{.dimensions = {3, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  static float DEFAULT_XYZ_MULTIPLIER[]{1.0, -1.0, 1.0};
  auto pointXYZMultiplier = std::make_shared<PipelineTensor>(
      m_secureMrMap2dTo3dPipeline,
      TensorAttribute{.dimensions = {3, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_XYZ_MULTIPLIER), sizeof(DEFAULT_XYZ_MULTIPLIER));

  static float DEFAULT_OFFSET[]{0.1, 0.25, -0.05};
  auto offsetTensor = std::make_shared<PipelineTensor>(
      m_secureMrMap2dTo3dPipeline,
      TensorAttribute{.dimensions = {3, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_OFFSET), sizeof(DEFAULT_OFFSET));

  static float DEFAULT_R_VEC[]{0.0, 0.0, 0.0};
  auto rvecTensor = std::make_shared<PipelineTensor>(
      m_secureMrMap2dTo3dPipeline,
      TensorAttribute{.dimensions = {3, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_R_VEC), sizeof(DEFAULT_R_VEC));

  static float DEFAULT_S_VEC[]{0.1, 0.1, 0.1};
  auto svecTensor = std::make_shared<PipelineTensor>(
      m_secureMrMap2dTo3dPipeline,
      TensorAttribute{.dimensions = {3, 1}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO},
      reinterpret_cast<int8_t*>(DEFAULT_S_VEC), sizeof(DEFAULT_S_VEC));

  auto leftEyeTransform = std::make_shared<PipelineTensor>(
      m_secureMrMap2dTo3dPipeline,
      TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  // Step 3: Assembly!
  (*m_secureMrMap2dTo3dPipeline)
      .uv2Cam(uvPlaceholder1, timestampPlaceholder1, cameraMatrixPlaceholder1, leftImgePlaceholder,
              rightImagePlaceholder, pointXYZ)
      .elementwise(Pipeline::ElementwiseOp::MULTIPLY, {pointXYZ, pointXYZMultiplier}, pointXYZ)
      .arithmetic("({0} + {1})", {pointXYZ, offsetTensor}, pointXYZ)
      .transform(rvecTensor, pointXYZ, svecTensor, currentPositionPlaceholder)
      .camSpace2XrLocal(timestampPlaceholder1, nullptr, leftEyeTransform)
      .arithmetic("({0} * {1})", {leftEyeTransform, currentPositionPlaceholder}, currentPositionPlaceholder);
}

void FaceTracker::CreateSecureMrRenderingPipeline() {
  m_secureMrRenderingPipeline = std::make_shared<Pipeline>(frameworkSession);

  // Step 1: placeholders
  previousPositionPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, previousPositionGlobal);
  currentPositionPlaceholder1 =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, currentPositionGlobal);

  // Step 2: local tensors
  auto interpolatedResult = std::make_shared<PipelineTensor>(
      m_secureMrRenderingPipeline,
      TensorAttribute{.dimensions = {4, 4}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  gltfPlaceholderTensor = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, gltfAsset);

  static float DEFAULT_EYE_3x3_MAT[]{
      0.1f, 0.0f, 0.0f,  //
      0.0f, 0.1f, 0.0f,  //
      0.0f, 0.0f, 0.1f   //
  };
  auto R = std::make_shared<PipelineTensor>(
      m_secureMrRenderingPipeline,
      TensorAttribute{.dimensions = {3, 3}, .channels = 1, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  R->setData(reinterpret_cast<int8_t*>(DEFAULT_EYE_3x3_MAT), sizeof(DEFAULT_EYE_3x3_MAT));

  // Step 3: Assembly!
  (*m_secureMrRenderingPipeline)
      .arithmetic("({0} * 0.95 + {1} * 0.05)", {previousPositionPlaceholder, currentPositionPlaceholder1},
                  interpolatedResult)
      .assignment(interpolatedResult, previousPositionPlaceholder)
      .assignment(R, (*interpolatedResult)[{{0, 3}, {0, 3}}])
      .execRenderCommand(std::make_shared<RenderCommand_Render>(gltfPlaceholderTensor, interpolatedResult));
}

void FaceTracker::RunSecureMrVSTImagePipeline() {
  m_secureMrVSTImagePipeline->submit({{vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global},
                                      {vstOutputRightUint8Placeholder, vstOutputRightUint8Global},
                                      {vstTimestampPlaceholder, vstTimestampGlobal},
                                      {vstCameraMatrixPlaceholder, vstCameraMatrixGlobal},
                                      {vstOutputLeftFp32Placeholder, vstOutputLeftFp32Global}},
                                     XR_NULL_HANDLE, nullptr);
}

void FaceTracker::RunSecureMrModelInferencePipeline() {
  m_secureMrModelInferencePipeline->submit({{vstImagePlaceholder, vstOutputLeftFp32Global},
                                            {uvPlaceholder, uvGlobal},
                                            {isFaceDetectedPlaceholder, isFaceDetectedGlobal}},
                                           XR_NULL_HANDLE, nullptr);
}

void FaceTracker::RunSecureMrMap2dTo3dPipeline() {
  m_secureMrMap2dTo3dPipeline->submit({{uvPlaceholder1, uvGlobal},
                                       {timestampPlaceholder1, vstTimestampGlobal},
                                       {cameraMatrixPlaceholder1, vstCameraMatrixGlobal},
                                       {leftImgePlaceholder, vstOutputLeftUint8Global},
                                       {rightImagePlaceholder, vstOutputRightUint8Global},
                                       {currentPositionPlaceholder, currentPositionGlobal}},
                                      XR_NULL_HANDLE, nullptr);
}

void FaceTracker::RunSecureMrRenderingPipeline() {
  m_secureMrRenderingPipeline->submit({{previousPositionPlaceholder, previousPositionGlobal},
                                       {currentPositionPlaceholder1, currentPositionGlobal},
                                       {gltfPlaceholderTensor, gltfAsset}},
                                      XR_NULL_HANDLE, isFaceDetectedGlobal);
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<FaceTracker>(instance, session);
}
}  // namespace SecureMR
