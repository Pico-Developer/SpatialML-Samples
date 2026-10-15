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

#define NUMBER_OF_OBJECTS 10
#define PRE_NMS_CANDIDATES 128

namespace {
constexpr auto kTargetFrameInterval = std::chrono::milliseconds(100);
constexpr int kLabelTextLength = 18;
constexpr float kMinLabelWidthPixels = 80.0f;
constexpr float kMaxLabelWidthPixels = 112.0f;
constexpr float kVisibleConfidenceThreshold = 0.50f;
constexpr bool kRequireConfirmedDetections = true;
constexpr int kConfirmationFrames = 3;
constexpr float kConfirmationMatchIou = 0.40f;
constexpr uint64_t kLabelRefreshIntervalFrames = 15;
constexpr uint64_t kStartupLabelRefreshFrames = kConfirmationFrames;
constexpr std::array<std::array<uint8_t, 4>, NUMBER_OF_OBJECTS> kDetectionColors{{
    {255, 90, 95, 255},    // coral red
    {0, 212, 255, 255},    // cyan
    {124, 252, 0, 255},    // lime
    {255, 215, 0, 255},    // yellow
    {255, 77, 255, 255},   // magenta
    {255, 140, 0, 255},    // orange
    {77, 124, 254, 255},   // blue
    {0, 230, 118, 255},    // green
    {255, 111, 181, 255},  // pink
    {179, 136, 255, 255},  // purple
}};
}  // namespace

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
  {
    std::lock_guard<std::mutex> guard(initialized_mtx);
    keepRunning = false;
  }
  initialized.notify_all();
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
    CreateSecureMrPostprocessPipeline();
    CreateSecureMrNmsPipeline();
    CreateSecureMrStabilizationPipeline();
    CreateSecureMrMap2dTo3dPipeline();
    CreateSecureMrLabelPipeline();
    CreateSecureMrRenderingPipeline();

    {
      std::lock_guard<std::mutex> guard(initialized_mtx);
      pipelineAllInitialized = true;
    }
    initialized.notify_all();
  });
}

void YoloDetector::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard, [this]() { return pipelineAllInitialized || !keepRunning; });
    }
    auto nextFrame = std::chrono::steady_clock::now();
    XrSecureMrPipelineRunPICO previousDetection = XR_NULL_HANDLE;
    uint64_t frameIndex = 0;
    while (keepRunning) {
      const auto vst = RunSecureMrVSTImagePipeline(previousDetection);
      const auto inference = RunSecureMrModelInferencePipeline(vst);
      const auto postprocess = RunSecureMrPostprocessPipeline(inference);
      const auto nms = RunSecureMrNmsPipeline(postprocess);
      const auto stabilized = RunSecureMrStabilizationPipeline(nms);
      const auto mapping = RunSecureMrMap2dTo3dPipeline(stabilized);
      const bool refreshStartupLabel = frameIndex < kStartupLabelRefreshFrames;
      const bool refreshPeriodicLabel = frameIndex % kLabelRefreshIntervalFrames == 0;
      if (refreshStartupLabel || refreshPeriodicLabel) {
        previousDetection = RunSecureMrLabelPipeline(mapping);
        RunSecureMrRenderingPipeline(previousDetection);
      } else {
        previousDetection = mapping;
        RunSecureMrRenderingPipeline(mapping);
      }
      ++frameIndex;

      nextFrame += kTargetFrameInterval;
      if (nextFrame < std::chrono::steady_clock::now()) {
        nextFrame = std::chrono::steady_clock::now();
      }
      std::this_thread::sleep_until(nextFrame);
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

void YoloDetector::DrawLabel(const std::shared_ptr<Pipeline>& pipeline,
                             const std::shared_ptr<PipelineTensor>& text,
                             const std::shared_ptr<PipelineTensor>& gltfPlaceholder,
                             const uint16_t textureId,
                             const std::array<uint8_t, 4>& color) {
  auto startTensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 2,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto colorTensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {2},
                                .channels = 4,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_COLOR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
  auto textureIDTensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO});
  auto fontSizeTensor = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  float startData[2]{0.03f, 0.65f};
  uint8_t colorData[8]{color[0], color[1], color[2], color[3], 24, 24, 24, 224};
  uint16_t textureIdData = textureId;
  float fontSize = 46.0f;
  startTensor->setData(reinterpret_cast<int8_t*>(startData), sizeof(startData));
  colorTensor->setData(reinterpret_cast<int8_t*>(colorData), sizeof(colorData));
  textureIDTensor->setData(reinterpret_cast<int8_t*>(&textureIdData), sizeof(textureIdData));
  fontSizeTensor->setData(reinterpret_cast<int8_t*>(&fontSize), sizeof(fontSize));

  auto drawText = std::make_shared<RenderCommand_DrawText>();
  drawText->text = text;
  drawText->startPosition = startTensor;
  drawText->fontSize = fontSizeTensor;
  drawText->colors = colorTensor;
  drawText->textureId = textureIDTensor;
  drawText->typeFace = RenderCommand_DrawText::TypeFaceTypes::BOLD;
  drawText->canvasWidth = 512;
  drawText->canvasHeight = 128;
  drawText->languageAndLocale = "en-US";
  drawText->gltfTensor = gltfPlaceholder;
  (*pipeline).execRenderCommand(drawText);
}

std::shared_ptr<PipelineTensor> YoloDetector::CreateDetectionPose(
    const std::shared_ptr<Pipeline>& pipeline,
    const std::shared_ptr<PipelineTensor>& pointXYZ,
    const std::shared_ptr<PipelineTensor>& box,
    const std::shared_ptr<PipelineTensor>& cameraMatrix,
    const std::shared_ptr<PipelineTensor>& score,
    const std::shared_ptr<PipelineTensor>& allBoxes,
    const std::shared_ptr<PipelineTensor>& allScores,
    const bool label,
    const int labelSlot) {

  auto rvecTensor = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                   .channels = 1,
                                                                                                   .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                   .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  // Frame and label geometry are generated in the native VST camera basis, so
  // neither requires a compensating in-plane rotation at runtime.
  float rvecData[3]{0.0f, 0.0f, 0.0f};
  rvecTensor->setData(reinterpret_cast<int8_t*>(rvecData), sizeof(rvecData));

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


  const auto makeScalar = [&pipeline]() {
    return std::make_shared<PipelineTensor>(
        pipeline, TensorAttribute{.dimensions = {1, 1},
                                  .channels = 1,
                                  .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  };

  auto depth = makeScalar();
  auto minDepth = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 1},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  float minDepthData = -1.5f;
  minDepth->setData(reinterpret_cast<int8_t*>(&minDepthData), sizeof(minDepthData));

  auto renderDepth = makeScalar();
  auto positiveDepth = makeScalar();
  auto depthRatio = makeScalar();
  auto pointXYZAdj = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                    .channels = 1,
                                                                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  auto viewOffset = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                      .channels = 1,
                                                                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float viewOffsetData[3]{0.1f, 0.0f, 0.0f};
  viewOffset->setData(reinterpret_cast<int8_t*>(viewOffsetData), sizeof(viewOffsetData));

  auto boxMin = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1, 2},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto boxMax = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1, 2},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto boxSize = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1, 2},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto boxWidth = makeScalar();
  auto boxHeight = makeScalar();
  auto focalX = makeScalar();
  auto focalY = makeScalar();
  auto scaleX = makeScalar();
  auto scaleY = makeScalar();
  auto scale = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {3, 1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float scaleData[3]{1.0f, 1.0f, label ? 0.01f : 0.05f};
  scale->setData(reinterpret_cast<int8_t*>(scaleData), sizeof(scaleData));

  (*pipeline).assignment((*pointXYZ)[{{0, 1},{2, 3}}], depth)
      .elementwise(Pipeline::ElementwiseOp::MIN, {depth, minDepth}, renderDepth)
      .arithmetic("({0} / {1})", {renderDepth, depth}, depthRatio)
      .assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{0, 1}}])
      .assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{1, 2}}])
      .assignment(depthRatio, (*pointXYZAdj)[{{0, 1},{2, 3}}])
      .elementwise(Pipeline::ElementwiseOp::MULTIPLY, {pointXYZ, pointXYZAdj}, pointXYZ)
      .arithmetic("(0.0 - {0})", {renderDepth}, positiveDepth)
      .assignment((*box)[{{0, 1}, {0, 2}}], boxMin)
      .assignment((*box)[{{0, 1}, {2, 4}}], boxMax)
      .arithmetic("({0} - {1})", {boxMax, boxMin}, boxSize)
      .assignment((*boxSize)[{{0, 1}, {0, 1}}], boxWidth)
      .assignment((*boxSize)[{{0, 1}, {1, 2}}], boxHeight)
      .assignment((*cameraMatrix)[{{0, 1}, {0, 1}}], focalX)
      .assignment((*cameraMatrix)[{{1, 2}, {1, 2}}], focalY);

  if (label) {
    const auto makeBoolScalar = [&pipeline]() {
      return std::make_shared<PipelineTensor>(
          pipeline, TensorAttribute{.dimensions = {1, 1},
                                    .channels = 1,
                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
    };

    auto labelWidth = makeScalar();
    auto labelHeight = makeScalar();
    auto labelWidthAtLeastMinimum = makeScalar();
    auto minLabelWidth = makeScalar();
    auto maxLabelWidth = makeScalar();
    auto horizontalAnchorSign = makeScalar();
    auto verticalAnchorSign = makeScalar();
    auto boxMinX = makeScalar();
    auto boxMinY = makeScalar();
    auto boxMaxX = makeScalar();
    auto boxCenterX = makeScalar();
    auto labelCenterX = makeScalar();
    auto currentLabelMinX = makeScalar();
    auto currentLabelMaxX = makeScalar();
    auto currentLabelMinY = makeScalar();
    auto currentLabelMaxY = makeScalar();
    auto deltaU = makeScalar();
    auto deltaV = makeScalar();
    auto deltaX = makeScalar();
    auto deltaY = makeScalar();
    float minLabelWidthData = kMinLabelWidthPixels;
    float maxLabelWidthData = kMaxLabelWidthPixels;
    float horizontalAnchorSignData = labelSlot % 2 == 0 ? 1.0f : -1.0f;
    minLabelWidth->setData(reinterpret_cast<int8_t*>(&minLabelWidthData), sizeof(minLabelWidthData));
    maxLabelWidth->setData(reinterpret_cast<int8_t*>(&maxLabelWidthData), sizeof(maxLabelWidthData));
    horizontalAnchorSign->setData(reinterpret_cast<int8_t*>(&horizontalAnchorSignData),
                                  sizeof(horizontalAnchorSignData));

    auto labelOffset = std::make_shared<PipelineTensor>(
        pipeline, TensorAttribute{.dimensions = {1, 3},
                                  .channels = 1,
                                  .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                  .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    float labelOffsetData[3]{0.0f, 0.0f, 0.02f};
    labelOffset->setData(reinterpret_cast<int8_t*>(labelOffsetData), sizeof(labelOffsetData));

    (*pipeline)
        .elementwise(Pipeline::ElementwiseOp::MAX, {boxWidth, minLabelWidth}, labelWidthAtLeastMinimum)
        .elementwise(Pipeline::ElementwiseOp::MIN, {labelWidthAtLeastMinimum, maxLabelWidth}, labelWidth)
        .arithmetic("({0} * 0.25)", {labelWidth}, labelHeight)
        .assignment((*boxMin)[{{0, 1}, {0, 1}}], boxMinX)
        .assignment((*boxMin)[{{0, 1}, {1, 2}}], boxMinY)
        .assignment((*boxMax)[{{0, 1}, {0, 1}}], boxMaxX)
        .arithmetic("(({0} + {1}) * 0.5)", {boxMinX, boxMaxX}, boxCenterX)
        .arithmetic("(({0} - {1}) * 0.5 * {2})",
                    {labelWidth, boxWidth, horizontalAnchorSign}, deltaU)
        .arithmetic("({0} + {1})", {boxCenterX, deltaU}, labelCenterX)
        .arithmetic("({0} - {1} * 0.5)", {labelCenterX, labelWidth}, currentLabelMinX)
        .arithmetic("({0} + {1} * 0.5)", {labelCenterX, labelWidth}, currentLabelMaxX)
        .assignment(boxMinY, currentLabelMinY)
        .arithmetic("({0} + {1})", {boxMinY, labelHeight}, currentLabelMaxY);

    auto anyCollision = makeBoolScalar();
    int32_t noCollisionData = 0;
    anyCollision->setData(reinterpret_cast<int8_t*>(&noCollisionData), sizeof(noCollisionData));

    auto visibilityThreshold = makeScalar();
    float visibilityThresholdData = kVisibleConfidenceThreshold;
    visibilityThreshold->setData(reinterpret_cast<int8_t*>(&visibilityThresholdData),
                                 sizeof(visibilityThresholdData));

    // NMS outputs are confidence-sorted. Keep the higher-confidence label on
    // top; if this preferred top rectangle intersects an earlier visible top
    // rectangle, move the current (lower-confidence) label to the bottom.
    for (int previous = 0; previous < labelSlot; ++previous) {
      auto previousBox = std::make_shared<PipelineTensor>(
          pipeline, TensorAttribute{.dimensions = {1, 4},
                                    .channels = 1,
                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
      auto previousBoxMin = std::make_shared<PipelineTensor>(
          pipeline, TensorAttribute{.dimensions = {1, 2},
                                    .channels = 1,
                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
      auto previousBoxMax = std::make_shared<PipelineTensor>(
          pipeline, TensorAttribute{.dimensions = {1, 2},
                                    .channels = 1,
                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
      auto previousMinX = makeScalar();
      auto previousMinY = makeScalar();
      auto previousMaxX = makeScalar();
      auto previousWidth = makeScalar();
      auto previousWidthAtLeastMinimum = makeScalar();
      auto previousLabelWidth = makeScalar();
      auto previousLabelHeight = makeScalar();
      auto previousHorizontalSign = makeScalar();
      auto previousDeltaU = makeScalar();
      auto previousCenterX = makeScalar();
      auto previousLabelMinX = makeScalar();
      auto previousLabelMaxX = makeScalar();
      auto previousLabelMaxY = makeScalar();
      auto previousScore = makeScalar();

      float previousHorizontalSignData = previous % 2 == 0 ? 1.0f : -1.0f;
      previousHorizontalSign->setData(reinterpret_cast<int8_t*>(&previousHorizontalSignData),
                                      sizeof(previousHorizontalSignData));

      (*pipeline)
          .assignment((*allBoxes)[{{previous, previous + 1}, {0, -1}}], previousBox)
          .assignment((*previousBox)[{{0, 1}, {0, 2}}], previousBoxMin)
          .assignment((*previousBox)[{{0, 1}, {2, 4}}], previousBoxMax)
          .assignment((*previousBoxMin)[{{0, 1}, {0, 1}}], previousMinX)
          .assignment((*previousBoxMin)[{{0, 1}, {1, 2}}], previousMinY)
          .assignment((*previousBoxMax)[{{0, 1}, {0, 1}}], previousMaxX)
          .arithmetic("({0} - {1})", {previousMaxX, previousMinX}, previousWidth)
          .elementwise(Pipeline::ElementwiseOp::MAX,
                       {previousWidth, minLabelWidth}, previousWidthAtLeastMinimum)
          .elementwise(Pipeline::ElementwiseOp::MIN,
                       {previousWidthAtLeastMinimum, maxLabelWidth}, previousLabelWidth)
          .arithmetic("({0} * 0.25)", {previousLabelWidth}, previousLabelHeight)
          .arithmetic("(({0} - {1}) * 0.5 * {2})",
                      {previousLabelWidth, previousWidth, previousHorizontalSign}, previousDeltaU)
          .arithmetic("(({0} + {1}) * 0.5 + {2})",
                      {previousMinX, previousMaxX, previousDeltaU}, previousCenterX)
          .arithmetic("({0} - {1} * 0.5)",
                      {previousCenterX, previousLabelWidth}, previousLabelMinX)
          .arithmetic("({0} + {1} * 0.5)",
                      {previousCenterX, previousLabelWidth}, previousLabelMaxX)
          .arithmetic("({0} + {1})", {previousMinY, previousLabelHeight}, previousLabelMaxY)
          .assignment((*allScores)[{{previous, previous + 1}, {0, 1}}], previousScore);

      auto overlapsLeft = makeBoolScalar();
      auto overlapsRight = makeBoolScalar();
      auto overlapsAbove = makeBoolScalar();
      auto overlapsBelow = makeBoolScalar();
      auto overlapsX = makeBoolScalar();
      auto overlapsY = makeBoolScalar();
      auto overlapsRect = makeBoolScalar();
      auto previousVisible = makeBoolScalar();
      auto visibleOverlap = makeBoolScalar();
      auto accumulatedCollision = makeBoolScalar();

      (*pipeline)
          .compareTo(*currentLabelMinX < previousLabelMaxX, overlapsLeft)
          .compareTo(*currentLabelMaxX > previousLabelMinX, overlapsRight)
          .compareTo(*currentLabelMinY < previousLabelMaxY, overlapsAbove)
          .compareTo(*currentLabelMaxY > previousMinY, overlapsBelow)
          .elementwise(Pipeline::ElementwiseOp::AND, {overlapsLeft, overlapsRight}, overlapsX)
          .elementwise(Pipeline::ElementwiseOp::AND, {overlapsAbove, overlapsBelow}, overlapsY)
          .elementwise(Pipeline::ElementwiseOp::AND, {overlapsX, overlapsY}, overlapsRect)
          .compareTo(*previousScore > visibilityThreshold, previousVisible)
          .elementwise(Pipeline::ElementwiseOp::AND,
                       {overlapsRect, previousVisible}, visibleOverlap)
          .elementwise(Pipeline::ElementwiseOp::OR,
                       {anyCollision, visibleOverlap}, accumulatedCollision);
      anyCollision = accumulatedCollision;
    }

    auto collisionFloat = makeScalar();
    (*pipeline)
        .assignment(anyCollision, collisionFloat)
        .arithmetic("(1.0 - {0} * 2.0)", {collisionFloat}, verticalAnchorSign)
        .arithmetic("(({0} - {1}) * 0.5 * {2})",
                    {labelHeight, boxHeight, verticalAnchorSign}, deltaV)
        .arithmetic("({0} * {1} / {2})", {positiveDepth, deltaU, focalX}, deltaX)
        .arithmetic("({0} * {1} / {2})", {positiveDepth, deltaV, focalY}, deltaY)
        .assignment(deltaX, (*labelOffset)[{{0, 1}, {0, 1}}])
        .assignment(deltaY, (*labelOffset)[{{0, 1}, {1, 2}}])
        .arithmetic("({0} + {1})", {pointXYZ, labelOffset}, pointXYZ)
        .arithmetic("({0} * {1} / {2})", {positiveDepth, labelWidth, focalX}, scaleX)
        .arithmetic("({0} * {1} / {2})", {positiveDepth, labelHeight, focalY}, scaleY);
  } else {
    (*pipeline)
        .arithmetic("({0} * {1} / ({2} * 10.0))", {positiveDepth, boxWidth, focalX}, scaleX)
        .arithmetic("({0} * {1} / ({2} * 10.0))", {positiveDepth, boxHeight, focalY}, scaleY);
  }
  (*pipeline)
      .assignment(scaleX, (*scale)[{{0, 1}, {0, 1}}])
      .assignment(scaleY, (*scale)[{{1, 2}, {0, 1}}]);

  auto isDetected = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  auto threshold = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float thresholdData = kVisibleConfidenceThreshold;
  threshold->setData(reinterpret_cast<int8_t*>(&thresholdData), sizeof(thresholdData));
  auto detectionScale = std::make_shared<PipelineTensor>(
      pipeline, TensorAttribute{.dimensions = {1, 1},
                                .channels = 1,
                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  (*pipeline).compareTo(*score > threshold, isDetected)
      .assignment(isDetected, detectionScale)
      .arithmetic("({0} * {1})", {scale, detectionScale}, scale)
      .elementwise(Pipeline::ElementwiseOp::MULTIPLY, {pointXYZ, multiplier}, pointXYZ)
      .arithmetic("({0} + {1})", {pointXYZ, viewOffset}, pointXYZ)
      .transform(rvecTensor, pointXYZ, scale, currentPosition)
      .camSpace2XrLocal(timestampPlaceholder2, nullptr, leftEyeTransform)
      .arithmetic("({0} * {1})", {leftEyeTransform, currentPosition}, currentPosition);

  // Keep both boxes and their attached labels on the same fixed world axes.
  // Position still follows the detected box, but label orientation no longer
  // inherits headset roll from the camera-to-local transform.
  auto scale_y = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 1},
                                                                                                .channels = 1,
                                                                                                .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  (*pipeline).assignment((*currentPosition)[{{1, 2}, {1, 2}}], scale_y);

  auto override_col = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {1, 3},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float overrideColData[3]{0.0f, 0.01f, 0.0f};
  override_col->setData(reinterpret_cast<int8_t*>(overrideColData), sizeof(overrideColData));
  auto override_row = std::make_shared<PipelineTensor>(pipeline, TensorAttribute{.dimensions = {3, 1},
                                                                                                     .channels = 1,
                                                                                                     .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                                                                     .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float overrideRowData[3]{0.0f, 0.01f, 0.0f};
  override_row->setData(reinterpret_cast<int8_t*>(overrideRowData), sizeof(overrideRowData));
  (*pipeline).assignment(override_row, (*currentPosition)[{{0, 3}, {1, 2}}])
      .assignment(override_col, (*currentPosition)[{{1, 2}, {0, 3}}])
      .assignment(scale_y, (*currentPosition)[{{1, 2}, {1, 2}}]);

  return currentPosition;
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
  vstOutputLeftFp32Global = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {640, 640},
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
  vstOutputLeftFp32Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrVSTImagePipeline, vstOutputLeftFp32Global);

  (*m_secureMrVSTImagePipeline).cameraAccess(vstOutputLeftUint8Placeholder,
                                             vstOutputRightUint8Placeholder,
                                             vstTimestampPlaceholder,
                                             vstCameraMatrixPlaceholder)
      .assignment(vstOutputLeftUint8Placeholder, vstOutputLeftFp32Placeholder);
}

void YoloDetector::CreateSecureMrModelInferencePipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrModelInferencePipeline");

  m_secureMrModelInferencePipeline = std::make_shared<Pipeline>(frameworkSession);
  vstImagePlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, vstOutputLeftFp32Global);

  rawOutputGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {8400, 85},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  rawOutputPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrModelInferencePipeline, rawOutputGlobal);

  std::vector<char> modelData;
  if (LoadModelData(YOLO_MODEL_PATH, modelData)) {
    auto algPackageBuf = modelData.data();
    auto algPackageSize = modelData.size();

    std::vector<std::pair<std::string, std::shared_ptr<PipelineTensor>>> algOps{{"images", vstImagePlaceholder}};

    std::unordered_map<std::string, std::string> operandAliasing;
    operandAliasing["images"] = "images";

    std::vector<std::pair<std::string, std::shared_ptr<PipelineTensor>>> algResults{{"output0", rawOutputPlaceholder}};

    std::unordered_map<std::string, std::string> resultAliasing;
    resultAliasing["output0"] = "output0";

    (*m_secureMrModelInferencePipeline).runAlgorithmOrdered(
        algPackageBuf, algPackageSize, algOps, operandAliasing, algResults, resultAliasing, "yolox_m",
        XR_SECURE_MR_MODEL_TYPE_LITE_RT_MODEL_PICO, XR_SECURE_MR_MODEL_TARGET_NPU_PICO);
  } else {
    Log::Write(Log::Level::Error, "Failed to load model data from file.");
  }
}

void YoloDetector::CreateSecureMrPostprocessPipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrPostprocessPipeline");

  m_secureMrPostprocessPipeline = std::make_shared<Pipeline>(frameworkSession);
  rawOutputPostprocessPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrPostprocessPipeline, rawOutputGlobal);

  candidateScoresGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {PRE_NMS_CANDIDATES, 1},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  candidateBoxesGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {PRE_NMS_CANDIDATES, 4},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  candidateClassesGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {PRE_NMS_CANDIDATES, 1},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});

  candidateScoresPostprocessPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrPostprocessPipeline, candidateScoresGlobal);
  candidateBoxesPostprocessPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrPostprocessPipeline, candidateBoxesGlobal);
  candidateClassesPostprocessPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrPostprocessPipeline, candidateClassesGlobal);

  std::vector<char> modelData;
  if (LoadModelData(YOLO_POSTPROCESS_MODEL_PATH, modelData)) {
    std::vector<std::pair<std::string, std::shared_ptr<PipelineTensor>>> inputs{
        {"predictions", rawOutputPostprocessPlaceholder}};
    // TFLite output tensor order is scores, boxes, then classes. Ordered binding
    // avoids relying on converter-generated PartitionedCall tensor names.
    std::vector<std::pair<std::string, std::shared_ptr<PipelineTensor>>> outputs{
        {"candidate_scores", candidateScoresPostprocessPlaceholder},
        {"candidate_boxes", candidateBoxesPostprocessPlaceholder},
        {"candidate_classes", candidateClassesPostprocessPlaceholder}};
    const std::unordered_map<std::string, std::string> noAliases;
    (*m_secureMrPostprocessPipeline)
        .runAlgorithmOrdered(modelData.data(), modelData.size(), inputs, noAliases, outputs, noAliases,
                             "yolox_postprocess", XR_SECURE_MR_MODEL_TYPE_LITE_RT_MODEL_PICO,
                             XR_SECURE_MR_MODEL_TARGET_CPU_PICO, 4);
  } else {
    Log::Write(Log::Level::Error, "Failed to load YOLOX post-processing model data.");
  }
}

void YoloDetector::CreateSecureMrNmsPipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrNmsPipeline");

  m_secureMrNmsPipeline = std::make_shared<Pipeline>(frameworkSession);
  candidateScoresNmsPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrNmsPipeline, candidateScoresGlobal);
  candidateBoxesNmsPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrNmsPipeline, candidateBoxesGlobal);
  candidateClassesNmsPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrNmsPipeline, candidateClassesGlobal);

  classesSelectGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  classesSelectPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrNmsPipeline, classesSelectGlobal);

  nmsBoxesGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 4},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  nmsBoxesPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrNmsPipeline, nmsBoxesGlobal);

  nmsScoresGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  nmsScoresPlaceholder = PipelineTensor::PipelinePlaceholderLike(m_secureMrNmsPipeline, nmsScoresGlobal);

  auto nmsIndices = std::make_shared<PipelineTensor>(
      m_secureMrNmsPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                             .channels = 1,
                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});

  // Offset boxes by class for NMS so two different classes at the same image
  // location can both survive. The original boxes are gathered after NMS.
  auto classIdsFloat = std::make_shared<PipelineTensor>(
      m_secureMrNmsPipeline, TensorAttribute{.dimensions = {PRE_NMS_CANDIDATES, 1},
                                             .channels = 1,
                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto classOffsets = std::make_shared<PipelineTensor>(
      m_secureMrNmsPipeline, TensorAttribute{.dimensions = {PRE_NMS_CANDIDATES, 4},
                                             .channels = 1,
                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto boxesForNms = std::make_shared<PipelineTensor>(
      m_secureMrNmsPipeline, TensorAttribute{.dimensions = {PRE_NMS_CANDIDATES, 4},
                                             .channels = 1,
                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  auto offsetNmsBoxes = std::make_shared<PipelineTensor>(
      m_secureMrNmsPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 4},
                                             .channels = 1,
                                             .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                             .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  (*m_secureMrNmsPipeline)
      .assignment(candidateClassesNmsPlaceholder, classIdsFloat)
      .assignment(classIdsFloat, (*classOffsets)[{{0, -1}, {0, 1}}])
      .assignment(classIdsFloat, (*classOffsets)[{{0, -1}, {1, 2}}])
      .assignment(classIdsFloat, (*classOffsets)[{{0, -1}, {2, 3}}])
      .assignment(classIdsFloat, (*classOffsets)[{{0, -1}, {3, 4}}])
      .arithmetic("({0} + {1} * 1000.0)", {candidateBoxesNmsPlaceholder, classOffsets}, boxesForNms)
      .nms(candidateScoresNmsPlaceholder, boxesForNms, nmsScoresPlaceholder,
           offsetNmsBoxes, nmsIndices, 0.5);
  CopyTensorBySlice(m_secureMrNmsPipeline, candidateBoxesNmsPlaceholder, nmsBoxesPlaceholder,
                    nmsIndices, NUMBER_OF_OBJECTS);
  CopyTensorBySlice(m_secureMrNmsPipeline, candidateClassesNmsPlaceholder, classesSelectPlaceholder,
                    nmsIndices, NUMBER_OF_OBJECTS);
}

void YoloDetector::CreateSecureMrStabilizationPipeline() {
  m_secureMrStabilizationPipeline = std::make_shared<Pipeline>(frameworkSession);

  nmsBoxesStabilizationPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, nmsBoxesGlobal);
  nmsScoresStabilizationPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, nmsScoresGlobal);
  classesStabilizationPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, classesSelectGlobal);

  confirmationStateGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 6},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  std::vector<float> initialState(NUMBER_OF_OBJECTS * 6, 0.0f);
  confirmationStateGlobal->setData(reinterpret_cast<int8_t*>(initialState.data()),
                                   initialState.size() * sizeof(float));
  confirmationStatePlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, confirmationStateGlobal);

  stabilizedBoxesGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 4},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  stabilizedScoresGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  stabilizedClassesGlobal = std::make_shared<GlobalTensor>(
      frameworkSession, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 1},
                                        .channels = 1,
                                        .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                        .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  stabilizedBoxesPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, stabilizedBoxesGlobal);
  stabilizedScoresPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, stabilizedScoresGlobal);
  stabilizedClassesPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrStabilizationPipeline, stabilizedClassesGlobal);

  auto nextState = std::make_shared<PipelineTensor>(
      m_secureMrStabilizationPipeline,
      TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, 6},
                      .channels = 1,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});

  std::string stabilizationScript =
      "var boxes;var scores;var classes;var state;var next_state;"
      "var stable_boxes;var stable_scores;var stable_classes;"
      "var require_confirmation=" + std::string(kRequireConfirmedDetections ? "true" : "false") + ";"
      "var confirmation_frames=" + std::to_string(kConfirmationFrames) + ";"
      "var match_iou=" + std::to_string(kConfirmationMatchIou) + ";"
      "var threshold=" + std::to_string(kVisibleConfidenceThreshold) + ";"
      "var used=[];"
      "for(var p=0;p<" + std::to_string(NUMBER_OF_OBJECTS) + ";p++)used[p]=false;"
      "for(var z=0;z<" + std::to_string(NUMBER_OF_OBJECTS) + ";z++){"
      "for(var c=0;c<6;c++)next_state[z*6+c]=0;"
      "for(var b=0;b<4;b++)stable_boxes[z*4+b]=0;"
      "stable_scores[z]=0;stable_classes[z]=-1;}"
      "for(var i=0;i<" + std::to_string(NUMBER_OF_OBJECTS) + ";i++){"
      "var score=scores[i];var class_id=classes[i];"
      "if(score<=threshold||class_id<0)continue;"
      "var best=-1;var best_iou=match_iou;"
      "var ax1=boxes[i*4];var ay1=boxes[i*4+1];"
      "var ax2=boxes[i*4+2];var ay2=boxes[i*4+3];"
      "for(var q=0;q<" + std::to_string(NUMBER_OF_OBJECTS) + ";q++){"
      "if(used[q]||state[q*6+5]<=0||state[q*6+4]!==class_id)continue;"
      "var bx1=state[q*6];var by1=state[q*6+1];"
      "var bx2=state[q*6+2];var by2=state[q*6+3];"
      "var ix1=Math.max(ax1,bx1);var iy1=Math.max(ay1,by1);"
      "var ix2=Math.min(ax2,bx2);var iy2=Math.min(ay2,by2);"
      "var iw=Math.max(0,ix2-ix1);var ih=Math.max(0,iy2-iy1);"
      "var intersection=iw*ih;"
      "var area_a=Math.max(0,ax2-ax1)*Math.max(0,ay2-ay1);"
      "var area_b=Math.max(0,bx2-bx1)*Math.max(0,by2-by1);"
      "var union_area=area_a+area_b-intersection;"
      "var overlap=union_area>0?intersection/union_area:0;"
      "if(overlap>best_iou){best_iou=overlap;best=q;}}"
      "var hits=1;if(best>=0){used[best]=true;hits=Math.min(confirmation_frames,state[best*6+5]+1);}"
      "for(var d=0;d<4;d++){next_state[i*6+d]=boxes[i*4+d];stable_boxes[i*4+d]=boxes[i*4+d];}"
      "next_state[i*6+4]=class_id;next_state[i*6+5]=hits;"
      "stable_classes[i]=class_id;"
      "stable_scores[i]=(!require_confirmation||hits>=confirmation_frames)?score:0;}";

  m_secureMrStabilizationPipeline->runJavascript(
      stabilizationScript.data(), stabilizationScript.size(),
      {{"boxes", nmsBoxesStabilizationPlaceholder},
       {"scores", nmsScoresStabilizationPlaceholder},
       {"classes", classesStabilizationPlaceholder},
       {"state", confirmationStatePlaceholder}},
      {{"next_state", nextState},
       {"stable_boxes", stabilizedBoxesPlaceholder},
       {"stable_scores", stabilizedScoresPlaceholder},
       {"stable_classes", stabilizedClassesPlaceholder}});
  (*m_secureMrStabilizationPipeline).assignment(nextState, confirmationStatePlaceholder);
}


void YoloDetector::CreateSecureMrMap2dTo3dPipeline() {

  m_secureMrMap2dTo3dPipeline = std::make_shared<Pipeline>(frameworkSession);

  nmsBoxesPlaceholder1 =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrMap2dTo3dPipeline, stabilizedBoxesGlobal);
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
}

void YoloDetector::CreateSecureMrLabelPipeline() {
  m_secureMrLabelPipeline = std::make_shared<Pipeline>(frameworkSession);
  classesSelectLabelPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrLabelPipeline, stabilizedClassesGlobal);
  nmsScoresLabelPlaceholder =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrLabelPipeline, stabilizedScoresGlobal);

  auto textToPrintTensor = std::make_shared<PipelineTensor>(
      m_secureMrLabelPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS, kLabelTextLength},
                                               .channels = 1,
                                               .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                               .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});

  std::string labelScript = "var class_ids;var scores;var label_texts;var labels=[";
  for (size_t classIndex = 0; classIndex < COCO_CLASSES.size(); ++classIndex) {
    if (classIndex != 0) labelScript += ',';
    labelScript += '\"' + COCO_CLASSES[classIndex] + '\"';
  }
  labelScript +=
      "];for(var i=0;i<" + std::to_string(NUMBER_OF_OBJECTS) + ";i++){"
      "var class_id=class_ids[i];"
      "var valid=class_id>=0&&class_id<labels.length&&class_id===Math.floor(class_id);"
      "var name=valid?labels[class_id]:\"unknown\";"
      "var confidence=Math.max(0,Math.min(100,Math.round(scores[i]*100.0)));"
      "var visible=valid&&scores[i]>" + std::to_string(kVisibleConfidenceThreshold) + ";"
      "var label=visible?name+\" \"+confidence+\"%\":\"\";"
      "for(var j=0;j<" + std::to_string(kLabelTextLength) + ";j++)"
      "label_texts[i*" + std::to_string(kLabelTextLength) + "+j]="
      "j<label.length?label.charCodeAt(j):32;}";
  m_secureMrLabelPipeline->runJavascript(
      labelScript.data(), labelScript.size(),
      {{"class_ids", classesSelectLabelPlaceholder}, {"scores", nmsScoresLabelPlaceholder}},
      {{"label_texts", textToPrintTensor}});

  std::vector<std::shared_ptr<PipelineTensor>> textTensors;
  textTensors.reserve(NUMBER_OF_OBJECTS);
  for (int index = 0; index < NUMBER_OF_OBJECTS; ++index) {
    auto text = std::make_shared<PipelineTensor>(
        m_secureMrLabelPipeline, TensorAttribute{.dimensions = {1, kLabelTextLength},
                                                 .channels = 1,
                                                 .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                 .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO});
    (*m_secureMrLabelPipeline).assignment((*textToPrintTensor)[{{index, index + 1}, {0, -1}}], text);
    textTensors.push_back(text);
  }

  std::vector<char> gltfData;
  if (LoadModelData(GLTF_PATH, gltfData)) {
    gltfAsset = std::make_shared<GlobalTensor>(frameworkSession, gltfData.data(), gltfData.size());
    gltfLabelPlaceholderTensor = PipelineTensor::PipelinePlaceholderLike(m_secureMrLabelPipeline, gltfAsset);
  } else {
    Log::Write(Log::Level::Error, "Failed to load glTF data from file.");
    return;
  }

  for (int index = 0; index < NUMBER_OF_OBJECTS; ++index) {
    DrawLabel(m_secureMrLabelPipeline, textTensors[index], gltfLabelPlaceholderTensor,
              static_cast<uint16_t>(index), kDetectionColors[index]);
  }
}

void YoloDetector::CreateSecureMrRenderingPipeline() {
  m_secureMrRenderingPipeline = std::make_shared<Pipeline>(frameworkSession);
  pointXYZPlaceholder1 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, pointXYZGlobal);
  timestampPlaceholder2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, vstTimestampGlobal);

  nmsBoxesPlaceholder2 =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, stabilizedBoxesGlobal);
  cameraMatrixPlaceholder2 = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, vstCameraMatrixGlobal);
  nmsScoresPlaceholder1 =
      PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, stabilizedScoresGlobal);

  gltfPlaceholderTensor = PipelineTensor::PipelinePlaceholderLike(m_secureMrRenderingPipeline, gltfAsset);

  auto nodePoses = std::make_shared<PipelineTensor>(
      m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {NUMBER_OF_OBJECTS * 2, 4, 4},
                                                    .channels = 1,
                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  std::vector<uint16_t> rootNodeIds;
  rootNodeIds.reserve(NUMBER_OF_OBJECTS * 2);
  for (int index = 0; index < NUMBER_OF_OBJECTS * 2; ++index) {
    rootNodeIds.push_back(static_cast<uint16_t>(index));
  }

  for (int index = 0; index < NUMBER_OF_OBJECTS; ++index) {
    auto point = std::make_shared<PipelineTensor>(
        m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1, 3},
                                                      .channels = 1,
                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    auto box = std::make_shared<PipelineTensor>(
        m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1, 4},
                                                      .channels = 1,
                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    auto score = std::make_shared<PipelineTensor>(
        m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1},
                                                      .channels = 1,
                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    (*m_secureMrRenderingPipeline)
        .assignment((*pointXYZPlaceholder1)[index], point)
        .assignment((*nmsBoxesPlaceholder2)[{{index, index + 1}, {0, -1}}], box)
        .assignment((*nmsScoresPlaceholder1)[{{index, index + 1}, {0, 1}}], score);
    auto framePose = CreateDetectionPose(
        m_secureMrRenderingPipeline, point, box, cameraMatrixPlaceholder2, score,
        nmsBoxesPlaceholder2, nmsScoresPlaceholder1, false, index);
    (*m_secureMrRenderingPipeline)
        .assignment(framePose, (*nodePoses)[{{index, index + 1}, {0, -1}, {0, -1}}]);

    auto labelPoint = std::make_shared<PipelineTensor>(
        m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1, 3},
                                                      .channels = 1,
                                                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
    (*m_secureMrRenderingPipeline).assignment((*pointXYZPlaceholder1)[index], labelPoint);
    auto labelPose = CreateDetectionPose(
        m_secureMrRenderingPipeline, labelPoint, box, cameraMatrixPlaceholder2, score,
        nmsBoxesPlaceholder2, nmsScoresPlaceholder1, true, index);
    (*m_secureMrRenderingPipeline)
        .assignment(labelPose,
                    (*nodePoses)[{{NUMBER_OF_OBJECTS + index, NUMBER_OF_OBJECTS + index + 1},
                                  {0, -1}, {0, -1}}]);
  }

  (*m_secureMrRenderingPipeline)
      .execRenderCommand(std::make_shared<RenderCommand_UpdateNodesLocalPoses>(
          gltfPlaceholderTensor, rootNodeIds, nodePoses));

  auto identity = std::make_shared<PipelineTensor>(
      m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {4, 4},
                                                    .channels = 1,
                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
  float identityData[16]{1.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 1.0f};
  identity->setData(reinterpret_cast<int8_t*>(identityData), sizeof(identityData));
  auto visible = std::make_shared<PipelineTensor>(
      m_secureMrRenderingPipeline, TensorAttribute{.dimensions = {1},
                                                    .channels = 1,
                                                    .usage = XR_SECURE_MR_TENSOR_TYPE_SCALAR_PICO,
                                                    .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO});
  int32_t visibleData = 1;
  visible->setData(reinterpret_cast<int8_t*>(&visibleData), sizeof(visibleData));
  auto render = std::make_shared<RenderCommand_Render>();
  render->gltfTensor = gltfPlaceholderTensor;
  render->pose = identity;
  render->visible = visible;
  (*m_secureMrRenderingPipeline).execRenderCommand(render);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrVSTImagePipeline(const XrSecureMrPipelineRunPICO waitFor) {
  return m_secureMrVSTImagePipeline->submit({{vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global},
                                             {vstOutputRightUint8Placeholder, vstOutputRightUint8Global},
                                             {vstTimestampPlaceholder, vstTimestampGlobal},
                                             {vstCameraMatrixPlaceholder, vstCameraMatrixGlobal},
                                             {vstOutputLeftFp32Placeholder, vstOutputLeftFp32Global}}, waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrModelInferencePipeline(
    const XrSecureMrPipelineRunPICO waitFor) {
  return m_secureMrModelInferencePipeline->submit({{vstImagePlaceholder, vstOutputLeftFp32Global},
                                                    {rawOutputPlaceholder, rawOutputGlobal}},
                                                   waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrPostprocessPipeline(
    const XrSecureMrPipelineRunPICO waitFor) {
  return m_secureMrPostprocessPipeline->submit(
      {{rawOutputPostprocessPlaceholder, rawOutputGlobal},
       {candidateScoresPostprocessPlaceholder, candidateScoresGlobal},
       {candidateBoxesPostprocessPlaceholder, candidateBoxesGlobal},
       {candidateClassesPostprocessPlaceholder, candidateClassesGlobal}},
      waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrNmsPipeline(const XrSecureMrPipelineRunPICO waitFor) {
  return m_secureMrNmsPipeline->submit(
      {{candidateScoresNmsPlaceholder, candidateScoresGlobal},
       {candidateBoxesNmsPlaceholder, candidateBoxesGlobal},
       {candidateClassesNmsPlaceholder, candidateClassesGlobal},
       {nmsBoxesPlaceholder, nmsBoxesGlobal},
       {nmsScoresPlaceholder, nmsScoresGlobal},
       {classesSelectPlaceholder, classesSelectGlobal}},
      waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrStabilizationPipeline(
    const XrSecureMrPipelineRunPICO waitFor) {
  return m_secureMrStabilizationPipeline->submit(
      {{nmsBoxesStabilizationPlaceholder, nmsBoxesGlobal},
       {nmsScoresStabilizationPlaceholder, nmsScoresGlobal},
       {classesStabilizationPlaceholder, classesSelectGlobal},
       {confirmationStatePlaceholder, confirmationStateGlobal},
       {stabilizedBoxesPlaceholder, stabilizedBoxesGlobal},
       {stabilizedScoresPlaceholder, stabilizedScoresGlobal},
       {stabilizedClassesPlaceholder, stabilizedClassesGlobal}},
      waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrMap2dTo3dPipeline(
    const XrSecureMrPipelineRunPICO waitFor) {
  return m_secureMrMap2dTo3dPipeline->submit({{nmsBoxesPlaceholder1, stabilizedBoxesGlobal},
                                              {timestampPlaceholder1, vstTimestampGlobal},
                                              {cameraMatrixPlaceholder1, vstCameraMatrixGlobal},
                                              {leftImgePlaceholder, vstOutputLeftUint8Global},
                                              {rightImagePlaceholder, vstOutputRightUint8Global},
                                              {pointXYZPlaceholder, pointXYZGlobal}}, waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrLabelPipeline(
    const XrSecureMrPipelineRunPICO waitFor) {
  std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> bindings{
      {classesSelectLabelPlaceholder, stabilizedClassesGlobal},
      {nmsScoresLabelPlaceholder, stabilizedScoresGlobal},
      {gltfLabelPlaceholderTensor, gltfAsset}};
  return m_secureMrLabelPipeline->submit(bindings, waitFor, nullptr);
}

XrSecureMrPipelineRunPICO YoloDetector::RunSecureMrRenderingPipeline(
    const XrSecureMrPipelineRunPICO waitFor) {
  std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> bindings{
      {pointXYZPlaceholder1, pointXYZGlobal},
      {timestampPlaceholder2, vstTimestampGlobal},
      {nmsBoxesPlaceholder2, stabilizedBoxesGlobal},
      {cameraMatrixPlaceholder2, vstCameraMatrixGlobal},
      {nmsScoresPlaceholder1, stabilizedScoresGlobal},
      {gltfPlaceholderTensor, gltfAsset}};
  return m_secureMrRenderingPipeline->submit(bindings, waitFor, nullptr);
}


std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<YoloDetector>(instance, session);
}
}  // namespace SecureMR
