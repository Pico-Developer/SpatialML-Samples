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

#include "yolo_object_detection.h"
#include "coco_classes.h"

#define NUMBER_OF_OBJECTS 3

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

YoloDetector::YoloDetector(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

YoloDetector::~YoloDetector() {
  keepRunning = false;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) runner.join();
  }
}

void YoloDetector::CreateFramework() {
  Log::Write(Log::Level::Info, "CreateFramework ...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, 640, 640);
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void YoloDetector::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {

    CreateSecureMrVSTImagePipeline();
    CreateSecureMrModelInferencePipeline();
    CreateSecureMrMap2dTo3dPipeline();
    CreateSecureMrRenderingPipeline();

    initialized.notify_all();
    pipelineAllInitialized = true;
  });
}

void YoloDetector::RunPipelines() {
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
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrMap2dTo3dPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrRenderingPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });
}

void YoloDetector::CopyTensorBySlice(const std::shared_ptr<Pipeline>& pipeline, const std::shared_ptr<PipelineTensor>& src,
                                        const std::shared_ptr<PipelineTensor>& dst, const std::shared_ptr<PipelineTensor>& indices, int32_t size) {
  // indices + 1
  auto indicesPlusOne = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                                                                                       .channels = 1,
                                                                                                       .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                       .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  (*pipeline).arithmetic("({0} + 1)", {indices}, indicesPlusOne);

  for (int i = 0; i < size; i++) {
    auto srcSlice = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {2},
                                                                                                   .channels = 2,
                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
    srcSlice->setData(reinterpret_cast<int8_t*>(new int32_t[4]{0, -1, 0, -1}), 4 * sizeof(int32_t));

    (*pipeline).assignment((*indices)[{{i, i+1}, {0, 1}}], (*srcSlice)[0][0])
        .assignment((*indicesPlusOne)[{{i, i+1}, {0, 1}}], (*srcSlice)[0][1])
        .assignment((*src)[srcSlice], (*dst)[{{i, i+1}, {0, -1}}]);
  }

}

void YoloDetector::CopyTextArray(const std::shared_ptr<Pipeline>& pipeline, std::vector<std::string>& textArray,
                                    const std::shared_ptr<PipelineTensor>& dstTensor) {

  for (int i  = 0; i < textArray.size(); i++) {
    int max_text_length = 13;
    auto textStr = textArray[i];
    std::string textWithSpaces = textStr;
    if (textStr.length() < max_text_length) {
      textWithSpaces.append(max_text_length - textStr.length(), ' ');
    }
    Log::Write(Log::Level::Info, "text: " + textWithSpaces);

    auto textTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {max_text_length},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});
    textTensor->setData(reinterpret_cast<int8_t*>(textWithSpaces.data()), max_text_length);
    (*pipeline).assignment(textTensor, (*dstTensor)[{{i, i+1}, {0, -1}}]);
  }
}

void YoloDetector::RenderText(const std::shared_ptr<Pipeline>& pipeline, const std::shared_ptr<PipelineTensor>& textArrayTensor,
                                 const std::shared_ptr<PipelineTensor>& pointXYZ, const std::shared_ptr<PipelineTensor>& gltfPlaceholder,
                                 const std::shared_ptr<PipelineTensor>& scale, const std::shared_ptr<PipelineTensor>& score) {

  auto rvecTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                   .channels = 1,
                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  rvecTensor->setData(reinterpret_cast<int8_t*>(new float[3]{0.0, 0.0, 0.0}), 3 * sizeof(float));

  auto leftEyeTransform = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {4, 4},
                                                                                                         .channels = 1,
                                                                                                         .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                         .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto currentPosition = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {4, 4},
                                                                                                        .channels = 1,
                                                                                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto multiplier = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                   .channels = 1,
                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  multiplier->setData(reinterpret_cast<int8_t*>(new float[3]{1.0, -1.0, 1.0}), 3 * sizeof(float));


  auto depth = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 1},
                                                                                              .channels = 1,
                                                                                              .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                              .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto minDepth = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 1},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  minDepth->setData(reinterpret_cast<int8_t*>(new float[1]{-1.5}), 1 * sizeof(float));

  auto depthRatio = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 1},
                                                                                                   .channels = 1,
                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto pointXYZAdj = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                    .channels = 1,
                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto offset = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                      .channels = 1,
                                                                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    offset->setData(reinterpret_cast<int8_t*>(new float[3]{0.1, 0.0, 0.0}), 3 * sizeof(float));

  (*pipeline).assignment((*pointXYZ)[{{0, 1},{2, 3}}], depth)
      .elementwise(Pipeline::ElementwiseOp::MIN, {depth, minDepth}, minDepth)
      .arithmetic("({0} / {1})", {minDepth, depth}, depthRatio)
      .assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{0, 1}}])
      .assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{1, 2}}])
      .assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{2, 3}}])
      .elementwise(Pipeline::ElementwiseOp::MULTIPLY, {pointXYZ, pointXYZAdj}, pointXYZ);

  (*pipeline).elementwise(Pipeline::ElementwiseOp::MULTIPLY, {pointXYZ, multiplier}, pointXYZ)
      .arithmetic("({0} + {1})", {pointXYZ, offset}, pointXYZ)
      .transform(rvecTensor, pointXYZ, scale, currentPosition)
      .camSpace2XrLocal(timestampPlaceholder2, nullptr, leftEyeTransform)
      .arithmetic("({0} * {1})", {leftEyeTransform, currentPosition}, currentPosition);

  // text render
  auto startTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1},
                                                                                                    .channels = 2,
                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto colorTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {2},
                                                                                                    .channels = 4,
                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  auto textureIDTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1},
                                                                                                        .channels = 1,
                                                                                                        .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
  auto fontSizeTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1},
                                                                                                       .channels = 1,
                                                                                                       .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                       .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  startTensor->setData(reinterpret_cast<int8_t*>(new float[2]{0.1, 0.5}), 2 * sizeof(float));
  colorTensor->setData(reinterpret_cast<int8_t*>(new uint8_t[8]{255, 255, 255, 255, 128, 128, 128, 128}), 8 * sizeof(uint8_t));
  textureIDTensor->setData(reinterpret_cast<int8_t*>(new uint16_t[1]{0}), 1 * sizeof(uint16_t));
  fontSizeTensor->setData(reinterpret_cast<int8_t*>(new float[1]{255.0}), 1 * sizeof(float));

  auto renderCommand_DrawText = std::make_shared<RenderCommand_DrawText>();
  renderCommand_DrawText->text = textArrayTensor;
  renderCommand_DrawText->startPosition = startTensor;
  renderCommand_DrawText->fontSize = fontSizeTensor;
  renderCommand_DrawText->colors = colorTensor;
  renderCommand_DrawText->textureId = textureIDTensor;
  renderCommand_DrawText->canvasWidth = 1024;
  renderCommand_DrawText->canvasHeight = 1024;
  renderCommand_DrawText->languageAndLocale = "en-US";
  renderCommand_DrawText->gltfTensor = gltfPlaceholder;
  (*pipeline).execRenderCommand(renderCommand_DrawText);

  auto isDetected = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1},
                                                                                                   .channels = 1,
                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto threshold = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1},
                                                                                                  .channels = 1,
                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  threshold->setData(reinterpret_cast<int8_t*>(new float[1]{0.6}), 1 * sizeof(float));
  (*pipeline).compareTo(*score > threshold, isDetected);

  auto scale_y = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 1},
                                                                                                .channels = 1,
                                                                                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  (*pipeline).assignment((*currentPosition)[{{1, 2}, {1, 2}}], scale_y);

  auto override_col = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  override_col->setData(reinterpret_cast<int8_t*>(new float[3]{0.0, 0.01, 0.0}), 3 * sizeof(float));
  auto override_row = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  override_row->setData(reinterpret_cast<int8_t*>(new float[3]{0.0, 0.01, 0.0}), 3 * sizeof(float));
  (*pipeline).assignment(override_row, (*currentPosition)[{{0, 3}, {1, 2}}])
      .assignment(override_col, (*currentPosition)[{{1, 2}, {0, 3}}])
      .assignment(scale_y, (*currentPosition)[{{1, 2}, {1, 2}}]);

  auto renderCommand_Render = std::make_shared<RenderCommand_Render>();
  renderCommand_Render->gltfTensor = gltfPlaceholder;
  renderCommand_Render->pose = currentPosition;
  renderCommand_Render->visible = isDetected;
  (*pipeline).execRenderCommand(renderCommand_Render);
}

void YoloDetector::CreateSecureMrVSTImagePipeline()  {
  Log::Write(Log::Level::Info, "SpatialML CreateSecureMrVSTImagePipeline");

  m_secureMrVSTImagePipeline = std::make_shared<Pipeline>(frameworkSession);

  vstOutputLeftUint8Global = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {640, 640},
                                                                                                                  .channels = 3,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  vstOutputRightUint8Global = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {640, 640},
                                                                                                                   .channels = 3,
                                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  vstOutputLeftFp32Global = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {640, 640},
                                                                                                                 .channels = 3,
                                                                                                                 .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                 .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  vstTimestampGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {1},
                                                                                                            .channels = 4,
                                                                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO,
                                                                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  vstCameraMatrixGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {3, 3},
                                                                                                               .channels = 1,
                                                                                                               .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                               .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  vstOutputLeftUint8Placeholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftUint8Global);
  vstOutputRightUint8Placeholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputRightUint8Global);
  vstTimestampPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstTimestampGlobal);
  vstCameraMatrixPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstCameraMatrixGlobal);

  vstOutputLeftFp32Placeholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftFp32Global);

  (*m_secureMrVSTImagePipeline).cameraAccess(vstOutputLeftUint8Placeholder,
                                             vstOutputRightUint8Placeholder,
                                             vstTimestampPlaceholder,
                                             vstCameraMatrixPlaceholder)
      .assignment(vstOutputLeftUint8Placeholder, vstOutputLeftFp32Placeholder)
      .arithmetic("{0} / 255.0", {vstOutputLeftFp32Placeholder}, vstOutputLeftFp32Placeholder);
}

void YoloDetector::CreateSecureMrModelInferencePipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrModelInferencePipeline");

  m_secureMrModelInferencePipeline = std::make_shared<Pipeline>(frameworkSession);
  vstImagePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, vstOutputLeftFp32Global);

  classesSelectGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                                                                                             .channels = 1,
                                                                                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  classesSelectPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, classesSelectGlobal);

  auto output = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 84},
                                                                                                                       .channels = 1,
                                                                                                                       .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                       .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  std::vector<char> modelData;
  if (LoadModelData(YOLO_MODEL_PATH, modelData)) {
    auto algPackageBuf = modelData.data();
    auto algPackageSize = modelData.size();

    std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> algOps;
    algOps["images"] = vstImagePlaceholder;

    std::unordered_map<std::string, std::string> operandAliasing;
    operandAliasing["images"] = "images";

    std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> algResults;
    algResults["output0"] = output;

    std::unordered_map<std::string, std::string> resultAliasing;
    resultAliasing["output0"] = "output0";

    (*m_secureMrModelInferencePipeline).runAlgorithm(algPackageBuf, algPackageSize, algOps, operandAliasing, algResults, resultAliasing, "yolo");
  } else {
    Log::Write(Log::Level::Error, "Failed to load model data from file.");
  }

  auto boxes = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 4},
                                                                                                                      .channels = 1,
                                                                                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto scores = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 80},
                                                                                                                       .channels = 1,
                                                                                                                       .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                       .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto xcenterycenter = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 2},
                                                                                                                               .channels = 1,
                                                                                                                               .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                               .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto widthHeight = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 2},
                                                                                                                            .channels = 1,
                                                                                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto xminymin = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 2},
                                                                                                                         .channels = 1,
                                                                                                                         .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                         .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto xmaxymax = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 2},
                                                                                                                         .channels = 1,
                                                                                                                         .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                         .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  (*m_secureMrModelInferencePipeline).assignment((*output)[{{0, 8400}, {0, 4}}], boxes)
      .assignment((*output)[{{0, 8400}, {4, 84}}], scores);

  (*m_secureMrModelInferencePipeline).assignment((*boxes)[{{0, 8400},{0, 2}}], xcenterycenter)
      .assignment((*boxes)[{{0, 8400},{2, 4}}], widthHeight)
      .arithmetic("({0} - {1} / 2)", {xcenterycenter, widthHeight}, xminymin)
      .arithmetic("({0} + {1} / 2)", {xcenterycenter, widthHeight}, xmaxymax)
      .assignment(xminymin, (*boxes)[{{0, 8400},{0, 2}}])
      .assignment(xmaxymax, (*boxes)[{{0, 8400},{2, 4}}]);

  auto sortedScores = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 80},
                                                                                                                             .channels = 1,
                                                                                                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto sortedIndicesPerRow = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 80},
                                                                                                                                    .channels = 1,
                                                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto bestScores = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 1},
                                                                                                                           .channels = 1,
                                                                                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto bestIndices = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {8400, 1},
                                                                                                                            .channels = 1,
                                                                                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});

  (*m_secureMrModelInferencePipeline).sortMatByRow(scores, sortedScores, sortedIndicesPerRow)
      .assignment((*sortedScores)[{{0, 8400}, {0, 1}}], bestScores)
      .assignment((*sortedIndicesPerRow)[{{0, 8400}, {0, 1}}], bestIndices);



  nmsBoxesGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 4},
                                                                                                        .channels = 1,
                                                                                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  nmsBoxesPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, nmsBoxesGlobal);

  nmsScoresGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                                                                                         .channels = 1,
                                                                                                         .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                         .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  nmsScoresPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, nmsScoresGlobal);

  auto nmsIndices = std::make_shared<PipelineTensor>(m_secureMrModelInferencePipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                                                                                                           .channels = 1,
                                                                                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});

  (*m_secureMrModelInferencePipeline).nms(bestScores, boxes, nmsScoresPlaceholder, nmsBoxesPlaceholder, nmsIndices, 0.5);

  CopyTensorBySlice(m_secureMrModelInferencePipeline, bestIndices, classesSelectPlaceholder, nmsIndices, NUMBER_OF_OBJECTS);
}


void YoloDetector::CreateSecureMrMap2dTo3dPipeline() {

  m_secureMrMap2dTo3dPipeline = std::make_shared<Pipeline>(frameworkSession);

  nmsBoxesPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, nmsBoxesGlobal);
  timestampPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstTimestampGlobal);
  cameraMatrixPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstCameraMatrixGlobal);
  leftImgePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstOutputLeftUint8Global);
  rightImagePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, vstOutputRightUint8Global);

  auto imagePoint = std::make_shared<PipelineTensor>(m_secureMrMap2dTo3dPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS},
                                                                                                                      .channels = 2,
                                                                                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                                                                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto xminymin = std::make_shared<PipelineTensor>(m_secureMrMap2dTo3dPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 2},
                                                                                                                    .channels = 1,
                                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto xmaxymax = std::make_shared<PipelineTensor>(m_secureMrMap2dTo3dPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 2},
                                                                                                                    .channels = 1,
                                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto imagePointMat = std::make_shared<PipelineTensor>(m_secureMrMap2dTo3dPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 2},
                                                                                                                         .channels = 1,
                                                                                                                         .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                         .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  (*m_secureMrMap2dTo3dPipeline).assignment((*nmsBoxesPlaceholder1)[{{0, -1}, {0, 2}}], xminymin)
      .assignment((*nmsBoxesPlaceholder1)[{{0, -1}, {2, 4}}], xmaxymax)
      .arithmetic("{0} * 0.5 + {1} * 0.5", {xminymin, xmaxymax}, imagePointMat)
      .assignment(imagePointMat, imagePoint);

  pointXYZGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS},
                                                                                                        .channels = 3,
                                                                                                        .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                                                                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  pointXYZPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, pointXYZGlobal);

  (*m_secureMrMap2dTo3dPipeline).uv2Cam(imagePoint, timestampPlaceholder1, cameraMatrixPlaceholder1, leftImgePlaceholder, rightImagePlaceholder, pointXYZPlaceholder);


  scaleGlobal = std::make_shared<GlobalTensor>(frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 3},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  scalePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, scaleGlobal);


  std::vector<float> scaleData(3 * NUMBER_OF_OBJECTS);
  for (int i = 0; i < NUMBER_OF_OBJECTS; ++i) {
    scaleData[3 * i] = 0.1f;
    scaleData[3 * i + 1] = 0.1f;
    scaleData[3 * i + 2] = 0.05f;
  }
  scaleGlobal->setData(reinterpret_cast<int8_t*>(scaleData.data()), 3 * NUMBER_OF_OBJECTS * sizeof(float));

  auto ratio = std::make_shared<PipelineTensor>(m_secureMrMap2dTo3dPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 2},
                                                                                                                 .channels = 1,
                                                                                                                 .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                 .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto divider = std::make_shared<PipelineTensor>(m_secureMrMap2dTo3dPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 2},
                                                                                                                   .channels = 1,
                                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  std::vector<float> dividerData(2 * NUMBER_OF_OBJECTS);
  for (int i = 0; i < NUMBER_OF_OBJECTS; ++i) {
    dividerData[2 * i] = 2000.0f;
    dividerData[2 * i + 1] = 2000.0f;
  }
  divider->setData(reinterpret_cast<int8_t*>(dividerData.data()), 2 * NUMBER_OF_OBJECTS * sizeof(float));

  (*m_secureMrMap2dTo3dPipeline).arithmetic("{0} - {1}", {xmaxymax, xminymin}, ratio)
      .arithmetic("{0} / {1}", {ratio, divider}, ratio);
  (*m_secureMrMap2dTo3dPipeline).assignment(ratio, (*scalePlaceholder)[{{0, -1}, {0, 2}}]);

}

void YoloDetector::CreateSecureMrRenderingPipeline() {

  m_secureMrRenderingPipeline = std::make_shared<Pipeline>(frameworkSession);

  pointXYZPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, pointXYZGlobal);
  timestampPlaceholder2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, vstTimestampGlobal);
  classesSelectPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, classesSelectGlobal);

  auto classesSelectInt = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                                                                                                            .channels = 1,
                                                                                                                            .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                            .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  (*m_secureMrRenderingPipeline).assignment(classesSelectPlaceholder1, classesSelectInt);

  auto textArrayTensor = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {80, 13},
                                                                                                                           .channels = 1,
                                                                                                                           .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                                           .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});
  auto textToPrintTensor = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 13},
                                                                                                                             .channels = 1,
                                                                                                                             .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});

  CopyTextArray(m_secureMrRenderingPipeline, COCO_CLASSES, textArrayTensor);

  CopyTensorBySlice(m_secureMrRenderingPipeline, textArrayTensor, textToPrintTensor, classesSelectInt, NUMBER_OF_OBJECTS);

  auto textArrayAttr = TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 13},
                                                 .channels = 1,
                                                 .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                 .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO};
  auto textArray0 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, textArrayAttr);
  auto textArray1 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, textArrayAttr);
  auto textArray2 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, textArrayAttr);

  auto pointXYZAttr = TensorAttribute{.dimensions = {1, 3},
                                                .channels = 1,
                                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO};
  auto pointXYZ0 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, pointXYZAttr);
  auto pointXYZ1 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, pointXYZAttr);
  auto pointXYZ2 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, pointXYZAttr);

  (*m_secureMrRenderingPipeline).assignment((*textToPrintTensor)[{{0, 1}, {0, -1}}], textArray0)
      .assignment((*textToPrintTensor)[{{1, 2}, {0, -1}}], textArray1)
      .assignment((*textToPrintTensor)[{{2, 3}, {0, -1}}], textArray2)
      .assignment((*pointXYZPlaceholder1)[0], pointXYZ0)
      .assignment((*pointXYZPlaceholder1)[1], pointXYZ1)
      .assignment((*pointXYZPlaceholder1)[2], pointXYZ2);

  std::vector<char> gltfData;
  if (LoadModelData(GLTF_PATH, gltfData)) {
    gltfAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfAsset1 = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfAsset2 = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfPlaceholderTensor = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, gltfAsset);
    gltfPlaceholderTensor1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, gltfAsset1);
    gltfPlaceholderTensor2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, gltfAsset2);

  } else {
    Log::Write(Log::Level::Error, "Failed to load glTF data from file.");
  }

  auto scale0 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                                  .channels = 1,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto scale1 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                                  .channels = 1,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto scale2 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                                  .channels = 1,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  scalePlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, scaleGlobal);

  (*m_secureMrRenderingPipeline).assignment((*scalePlaceholder1)[{{0, 1},{0, -1}}], scale0);
  (*m_secureMrRenderingPipeline).assignment((*scalePlaceholder1)[{{1, 2},{0, -1}}], scale1);
  (*m_secureMrRenderingPipeline).assignment((*scalePlaceholder1)[{{2, 3},{0, -1}}], scale2);

  nmsScoresPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, nmsScoresGlobal);
  auto score0 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1},
                                                                                                                  .channels = 1,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto score1 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1},
                                                                                                                  .channels = 1,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto score2 = std::make_shared<PipelineTensor>(m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1},
                                                                                                                  .channels = 1,
                                                                                                                  .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                                                                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  (*m_secureMrRenderingPipeline).assignment((*nmsScoresPlaceholder1)[{{0, 1},{0, 1}}], score0)
      .assignment((*nmsScoresPlaceholder1)[{{1, 2},{0, 1}}], score1)
      .assignment((*nmsScoresPlaceholder1)[{{2, 3},{0, 1}}], score2);

  RenderText(m_secureMrRenderingPipeline, textArray0, pointXYZ0, gltfPlaceholderTensor, scale0, score0);
  RenderText(m_secureMrRenderingPipeline, textArray1, pointXYZ1, gltfPlaceholderTensor1, scale1, score1);
  RenderText(m_secureMrRenderingPipeline, textArray2, pointXYZ2, gltfPlaceholderTensor2, scale2, score2);

}

void YoloDetector::RunSecureMrVSTImagePipeline() {
  m_secureMrVSTImagePipeline->submit({{vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global},
                                      {vstOutputRightUint8Placeholder, vstOutputRightUint8Global},
                                      {vstTimestampPlaceholder, vstTimestampGlobal},
                                      {vstCameraMatrixPlaceholder, vstCameraMatrixGlobal},
                                      {vstOutputLeftFp32Placeholder, vstOutputLeftFp32Global}}, XR_NULL_HANDLE, nullptr);
}

void YoloDetector::RunSecureMrModelInferencePipeline() {
  m_secureMrModelInferencePipeline->submit({{vstImagePlaceholder, vstOutputLeftFp32Global},
                                            {nmsBoxesPlaceholder, nmsBoxesGlobal},
                                            {nmsScoresPlaceholder, nmsScoresGlobal},
                                            {classesSelectPlaceholder, classesSelectGlobal}},
                                           XR_NULL_HANDLE, nullptr);
}

void YoloDetector::RunSecureMrMap2dTo3dPipeline() {
  m_secureMrMap2dTo3dPipeline->submit({{nmsBoxesPlaceholder1, nmsBoxesGlobal},
                                       {timestampPlaceholder1, vstTimestampGlobal},
                                       {cameraMatrixPlaceholder1, vstCameraMatrixGlobal},
                                       {leftImgePlaceholder, vstOutputLeftUint8Global},
                                       {rightImagePlaceholder, vstOutputRightUint8Global},
                                       {pointXYZPlaceholder, pointXYZGlobal},
                                       {scalePlaceholder, scaleGlobal}}, XR_NULL_HANDLE, nullptr);
}

void YoloDetector::RunSecureMrRenderingPipeline() {
  m_secureMrRenderingPipeline->submit({{gltfPlaceholderTensor, gltfAsset},
                                       {gltfPlaceholderTensor1, gltfAsset1},
                                       {gltfPlaceholderTensor2, gltfAsset2},
                                       {pointXYZPlaceholder1, pointXYZGlobal},
                                       {timestampPlaceholder2, vstTimestampGlobal},
                                       {classesSelectPlaceholder1, classesSelectGlobal},
                                       {nmsScoresPlaceholder1, nmsScoresGlobal},
                                       {scalePlaceholder1, scaleGlobal}}, XR_NULL_HANDLE, nullptr);
}


std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<YoloDetector>(instance, session);
}
}  // namespace SecureMR
