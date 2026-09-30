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

#pragma once
#include "pch.h"
#include <atomic>
#include <fstream>
#include <random>
#include "logger.h"
#include "common.h"
#include "securemr_base.h"
#include "securemr_utils/adapter.hpp"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/tensor.h"
#include "securemr_utils/rendercommand.h"
#include "securemr_utils/session.h"

#define YOLO_MODEL_PATH "yolom.tflite"
#define GLTF_PATH "frame2.gltf"

namespace SecureMR {

class YoloDetector : public ISecureMR {
 public:
  YoloDetector(const XrInstance& instance, const XrSession& session);

  ~YoloDetector() override;

  void CreateFramework() override;

  void CreatePipelines() override;

  void RunPipelines() override;

  [[nodiscard]] bool LoadingFinished() const override { return pipelineAllInitialized; }

 protected:
  void CreateGlobalTensor();

  void CreateSecureMrVSTImagePipeline();

  void CreateSecureMrModelInferencePipeline();

  void CreateSecureMrMap2dTo3dPipeline();

  void CreateSecureMrRenderingPipeline();

  void RunSecureMrVSTImagePipeline();

  void RunSecureMrModelInferencePipeline();

  void RunSecureMrMap2dTo3dPipeline();

  void RunSecureMrRenderingPipeline();

  static void CopyTensorBySlice(const std::shared_ptr<Pipeline>& pipeline, const std::shared_ptr<PipelineTensor>& src,
                                const std::shared_ptr<PipelineTensor>& dst, const std::shared_ptr<PipelineTensor>& indices, int32_t size);

  static void CopyTextArray(const std::shared_ptr<Pipeline>& pipeline, std::vector<std::string>& textArray,
                            const std::shared_ptr<PipelineTensor>& dstTensor);

  void RenderText(const std::shared_ptr<Pipeline>& pipeline, const std::shared_ptr<PipelineTensor>& textArray0,
                  const std::shared_ptr<PipelineTensor>& pointXYZ0, const std::shared_ptr<PipelineTensor>& gltfPlaceholder,
                  const std::shared_ptr<PipelineTensor>& scale, const std::shared_ptr<PipelineTensor>& score);


  XrInstance xr_instance;
  XrSession xr_session;

 private:
  // Root framework
  std::shared_ptr<FrameworkSession> frameworkSession;

  // Global tensors for pipeline communication
  // These tensors are shared between pipelines and serve as data transfer channels
  // They can also act as execution conditions for pipeline synchronization

  // Pipeline instances for different processing stages
  std::shared_ptr<Pipeline> m_secureMrVSTImagePipeline;        // Video See Through camera pipeline
  std::shared_ptr<Pipeline> m_secureMrModelInferencePipeline;  // YOLO model inference pipeline
  std::shared_ptr<Pipeline> m_secureMrMap2dTo3dPipeline;       // 2D to 3D coordinate mapping pipeline
  std::shared_ptr<Pipeline> m_secureMrRenderingPipeline;       // Result visualization pipeline

  // VST Pipeline IO
  // Handles stereo camera input processing and format conversion
  std::shared_ptr<GlobalTensor> vstOutputLeftUint8Global;      // Left camera frame in uint8 format
  std::shared_ptr<GlobalTensor> vstOutputRightUint8Global;     // Right camera frame in uint8 format
  std::shared_ptr<GlobalTensor> vstOutputLeftFp32Global;       // Left camera frame in float32 format for model input
  std::shared_ptr<GlobalTensor> vstTimestampGlobal;           // Camera frame timestamp for synchronization
  std::shared_ptr<GlobalTensor> vstCameraMatrixGlobal;        // Camera calibration matrix
  // Pipeline placeholders for VST tensors
  std::shared_ptr<PipelineTensor> vstOutputLeftUint8Placeholder;
  std::shared_ptr<PipelineTensor> vstOutputRightUint8Placeholder;
  std::shared_ptr<PipelineTensor> vstOutputLeftFp32Placeholder;
  std::shared_ptr<PipelineTensor> vstTimestampPlaceholder;
  std::shared_ptr<PipelineTensor> vstCameraMatrixPlaceholder;

  // Model Inference Pipeline IO
  // Handles YOLO model inference and detection results
  std::shared_ptr<GlobalTensor> classesSelectGlobal;          // Selected object classes from detection
  std::shared_ptr<GlobalTensor> nmsBoxesGlobal;              // Bounding boxes after NMS
  std::shared_ptr<GlobalTensor> nmsScoresGlobal;             // Confidence scores for detections
  // Pipeline placeholders for inference tensors
  std::shared_ptr<PipelineTensor> classesSelectPlaceholder;
  std::shared_ptr<PipelineTensor> nmsBoxesPlaceholder;
  std::shared_ptr<PipelineTensor> vstImagePlaceholder;
  std::shared_ptr<PipelineTensor> nmsScoresPlaceholder;

  // 2D to 3D Mapping Pipeline IO
  // Converts 2D detections to 3D world coordinates
  std::shared_ptr<GlobalTensor> pointXYZGlobal;              // 3D coordinates of detected objects
  std::shared_ptr<GlobalTensor> scaleGlobal;                // Scale information for visualization
  // Pipeline placeholders for mapping tensors
  std::shared_ptr<PipelineTensor> nmsBoxesPlaceholder1;
  std::shared_ptr<PipelineTensor> timestampPlaceholder1;
  std::shared_ptr<PipelineTensor> cameraMatrixPlaceholder1;
  std::shared_ptr<PipelineTensor> leftImgePlaceholder;
  std::shared_ptr<PipelineTensor> rightImagePlaceholder;
  std::shared_ptr<PipelineTensor> pointXYZPlaceholder;
  std::shared_ptr<PipelineTensor> scalePlaceholder;

  // Rendering Pipeline IO
  // Handles visualization of detection results
  std::shared_ptr<GlobalTensor> gltfAsset;                   // GLTF model assets for visualization
  std::shared_ptr<GlobalTensor> gltfAsset1;
  std::shared_ptr<GlobalTensor> gltfAsset2;
  // Pipeline placeholders for rendering tensors
  std::shared_ptr<PipelineTensor> gltfPlaceholderTensor;
  std::shared_ptr<PipelineTensor> gltfPlaceholderTensor1;
  std::shared_ptr<PipelineTensor> gltfPlaceholderTensor2;
  std::shared_ptr<PipelineTensor> pointXYZPlaceholder1;
  std::shared_ptr<PipelineTensor> timestampPlaceholder2;
  std::shared_ptr<PipelineTensor> classesSelectPlaceholder1;
  std::shared_ptr<PipelineTensor> scalePlaceholder1;
  std::shared_ptr<PipelineTensor> nmsScoresPlaceholder1;

  // Run-time control

  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;
  std::atomic<bool> keepRunning{true};
  std::atomic<bool> pipelineAllInitialized{false};
};

}  // namespace SecureMR
