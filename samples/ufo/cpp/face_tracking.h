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

#define FACE_DETECTION_MODEL_PATH "face_detector.tflite"
#define GLTF_PATH "UFO.gltf"
#define ANCHOR_MAT "anchors_1.mat"

namespace SecureMR {

class FaceTracker : public ISecureMR {
 public:
  FaceTracker(const XrInstance& instance, const XrSession& session);

  ~FaceTracker() override;

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

  XrInstance xr_instance;
  XrSession xr_session;

 private:
  // Root framework
  std::shared_ptr<FrameworkSession> frameworkSession;

  // Global tensors
  // Recall that global tensors are used to share data
  // between pipelines, and can also server as pipeline
  // execution condition

  /**
   * Caching the latest left-eye image --- shared between
   * the VST, the inference and the 2D-to-3D pipelines
   * <br/>
   * In R8G8B8 format
   */
  std::shared_ptr<GlobalTensor> vstOutputLeftUint8Global;
  /**
   * Caching the latest right-eye image --- shared between
   * the VST and the 2D-to-3D pipelines
   * <br/>
   * In R8G8B8 format
   */
  std::shared_ptr<GlobalTensor> vstOutputRightUint8Global;
  /**
   * Left-eye image, but converted to floating point, i.e.,
   * R32G32B32 format
   */
  std::shared_ptr<GlobalTensor> vstOutputLeftFp32Global;
  /**
   * Caching camera timestamp of the latest image --- shared between
   * the VST, the 2D-to-3D and the render pipelines, to
   * compensate the time different between camera exposure
   * and rendering
   */
  std::shared_ptr<GlobalTensor> vstTimestampGlobal;
  /**
   * Caching the camera matrix of the latest image --- shared
   * between the VST and the 2D-to-3D pipelines, to recover
   * the 3D coordinates from camera projection
   */
  std::shared_ptr<GlobalTensor> vstCameraMatrixGlobal;
  /**
   * The 2D points of the detected face key-points, i.e.,
   * the face detection result ---
   * shared between the inference and the 2D-to-3D pipelines
   */
  std::shared_ptr<GlobalTensor> uvGlobal;
  /**
   * A flag to determine whether the face detection model has
   * made a confident detection, used to determined whether
   * the render pipeline should be invoked based on the
   * outcome from the inference pipeline.
   */
  std::shared_ptr<GlobalTensor> isFaceDetectedGlobal;
  /**
   * The 3D pose from face detection --- shared between the
   * 2D-to-3D and the render pipelines
   */
  std::shared_ptr<GlobalTensor> currentPositionGlobal;
  /**
   * The previous 3D pose from face detection --- shared between
   * the 2D-to-3D and the render pipelines, to perform a
   * smooth movement of the UFO (glTF object to be rendered)
   */
  std::shared_ptr<GlobalTensor> previousPositionGlobal;
  /**
   * The glTF object to be rendered, a UFO in this demo
   * <br/>
   * </b>NOTE</b>Though the object is <i>only</i> used
   * by the render pipeline, but we require the glTF
   * object can only be created as a global tensor, as
   * the creation of glTF tensor is time-consuming and
   * associated with much resource.
   */
  std::shared_ptr<GlobalTensor> gltfAsset;

  // Pipelines, as computation graphs able to be individually
  // scheduled, consisting of operators as nodes and local tensors
  // as edges.

  /**
   * The VST pipeline, for camera access
   */
  std::shared_ptr<Pipeline> m_secureMrVSTImagePipeline;
  /**
   * The inference pipeline, where the face detection algorithm
   * is run, producing 2D key-points
   */
  std::shared_ptr<Pipeline> m_secureMrModelInferencePipeline;
  /**
   * The 2D-to-3D pipeline, for inverse projection to the 3D
   * pose of the detected face from the 2D key-points
   */
  std::shared_ptr<Pipeline> m_secureMrMap2dTo3dPipeline;
  /**
   * Render pipeline, where the animation is updated timely
   */
  std::shared_ptr<Pipeline> m_secureMrRenderingPipeline;

  // Placeholders for each pipeline
  // Recall placeholders are pipeline's local references to
  // global tensors, to avoid memory copy and competition
  // on the shared data between pipelines executed in different
  // threads.

  // Placeholders for the VST pipeline

  std::shared_ptr<PipelineTensor> vstOutputLeftUint8Placeholder;
  std::shared_ptr<PipelineTensor> vstOutputRightUint8Placeholder;
  std::shared_ptr<PipelineTensor> vstOutputLeftFp32Placeholder;
  std::shared_ptr<PipelineTensor> vstTimestampPlaceholder;
  std::shared_ptr<PipelineTensor> vstCameraMatrixPlaceholder;

  // Placeholders for the inference pipeline

  std::shared_ptr<PipelineTensor> vstImagePlaceholder;
  std::shared_ptr<PipelineTensor> uvPlaceholder;
  std::shared_ptr<PipelineTensor> isFaceDetectedPlaceholder;

  // Placeholders for the 2D-to-3D pipeline

  std::shared_ptr<PipelineTensor> uvPlaceholder1;
  std::shared_ptr<PipelineTensor> timestampPlaceholder1;
  std::shared_ptr<PipelineTensor> cameraMatrixPlaceholder1;
  std::shared_ptr<PipelineTensor> leftImgePlaceholder;
  std::shared_ptr<PipelineTensor> rightImagePlaceholder;
  std::shared_ptr<PipelineTensor> currentPositionPlaceholder;

  // PiPlaceholders for the render pipeline

  std::shared_ptr<PipelineTensor> gltfPlaceholderTensor;
  std::shared_ptr<PipelineTensor> previousPositionPlaceholder;
  std::shared_ptr<PipelineTensor> currentPositionPlaceholder1;

  // Run-time control

  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;
  std::atomic<bool> keepRunning{true};
  std::atomic<bool> pipelineAllInitialized{false};
};

}  // namespace SecureMR
