// Copyright (2026) Bytedance Ltd. and/or its affiliates
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
#include <functional>
#include <fstream>
#include <random>
#include <xr_linear.h>
#include "logger.h"
#include "common.h"
#include "securemr_base.h"
#include "securemr_utils/adapter.hpp"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/tensor.h"
#include "securemr_utils/rendercommand.h"
#include "securemr_utils/session.h"
#include <chrono>

#define WHACKAMOLE_DETECTION_MODEL_PATH "detection.tflite"
#define WHACKAMOLE_LANDMARK_MODEL_PATH "landmark.tflite"
#define JOINT_TRANSFORMER_JSCODE_PATH "jointsTransformer.js"
#define GLTF_PATH "pose_marker.gltf"
#define GLTF_SCOREBOARD_PATH "scoreboard_panel.gltf"
#define GLTF_DUCK_PATH "Duck.gltf"
#define GLTF_REDBOX_PATH "redbox.gltf"
#define ANCHOR_MAT "anchors_1.mat"
#define MAX_MOLES 5
#define MAX_JOINTS 4

namespace SecureMR {

class WhackamoleDetector : public ISecureMR {
 public:
  WhackamoleDetector(const XrInstance& instance, const XrSession& session);

  ~WhackamoleDetector() override;

  void CreateFramework() override;

  void CreatePipelines() override;

  void RunPipelines() override;

  [[nodiscard]] bool LoadingFinished() const override { return pipelineAllInitialized; }

  void UpdateHandPose(const XrVector3f* leftHandDelta, const XrVector3f* rightHandDelta) override;

 protected:
  /**
   * Create all the global tensors, must be called before create any pipelines
   */
  void CreateGlobalTensor();

  /**
   * Create the pipeline for retrieving RGB image
   */
  void CreateSecureMrVSTImagePipeline();

  /**
   * Create the pipeline<b>s</b> where the core logic of pose detection is running. There
   * are 3 bundled pipelines, and they shall be run in sequence:
   *
   * <ol>
   * <li><code>m_secureMrDetectionPipeline</code>, the pipeline where the detection
   *  algorithm is running, it intakes RGB image, determine the region which it believes
   *  contains a human body, and outputs a confidence score and an affine matrix from the
   *  RGB image to the region, in <code>roiAffineGlobal</code></li>
   * <li><code>m_secureMrAffineUpdatePipeline</code>, the pipeline to update the affine
   *  matrix in <code>roiAffineUpdatedGlobal</code> using <code>roiAffineGlobal</code>
   *  <i>only</i> if the detection score is high; otherwise, the old values in
   *  <code>roiAffineUpdatedGlobal</code> is retained.</li>
   * <li><code>m_secureMrLandmarkPipeline</code>, the pipeline where the pose-landmark
   *  model is running. It applies the affine matrix to get the image patch most likely
   *  containing the human body, runs the pose-landmark models and compute each-bone
   *  transforms for skeleton animation</li>
   * </ol>
   */
  void CreateSecureMrModelInferencePipeline();

  /**
   * Create the pipeline to update the world transform and each bone's local transform to drive the skeleton animation
   */
  void CreateSecureMrRenderingPipeline();

  /**
   * Submit the pipeline for retrieving RGB images for execution
   */
  XrSecureMrPipelineRunPICO RunSecureMrVSTImagePipeline(XrSecureMrPipelineRunPICO pre = XR_NULL_HANDLE);

  /**
   * Submit the pipeline for pose detection logic
   */
  XrSecureMrPipelineRunPICO RunSecureMrModelInferencePipeline(XrSecureMrPipelineRunPICO pre = XR_NULL_HANDLE);

  /**
   * Submit the pipeline for updating renderer
   */
  XrSecureMrPipelineRunPICO RunSecureMrRenderingPipeline(XrSecureMrPipelineRunPICO pre = XR_NULL_HANDLE);

  void CreateAndRunSecureMrMovePipeline(float x, float y, float z);
  void CreateAndRunInitMolesPipeline();

  XrInstance xr_instance;
  XrSession xr_session;

 private:
  /**
   *  Root framework
   */
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
   * Resized left-eye image, but converted to floating point, i.e.,
   * R32G32B32 format
   */
  std::shared_ptr<GlobalTensor> resizedLeftFp32Global;

  /**
   * A flag to determine whether the pose detection model has
   * made a confident detection, used to determined whether
   * the render pipeline should be invoked based on the
   * outcome from the inference pipeline.
   */
  std::shared_ptr<GlobalTensor> isPoseDetectedGlobal;

  /**
   * The affine transform from the raw camera image (512, 512)
   * to the ROI (region of interest) where the human body
   * is detected.
   */
  std::shared_ptr<GlobalTensor> roiAffineGlobal;

  /**
   * The affine transform can be unstable some time. In order to avoid
   * jittering, we discard the affine transform and reuse the previous
   * affine if the confidence for the new affine transform is low.
   * <br/>
   * This global tensor stores the affine transform from input camera
   * image to ROI <i>after</i> the rectification described above.
   **/
  std::shared_ptr<GlobalTensor> roiAffineUpdatedGlobal;


  std::shared_ptr<GlobalTensor> roiImageGlobal;

  /**
   * The global tensor stores skeleton animation's parameters
   * for current frame. It is of dimensions: (11, 4, 4), i.e.,
   * 11 4x4 transform matrices each corresponding to one of
   * the 11 bones in the glTF.
   **/
  std::shared_ptr<GlobalTensor> bodyLandmarkGlobal;
  std::shared_ptr<GlobalTensor> bodyLandmarkGlobal2;
  std::shared_ptr<GlobalTensor> molesTransformsGlobal;
  std::shared_ptr<GlobalTensor> moleCollisionsGlobal;
  std::shared_ptr<GlobalTensor> molesVisibleGlobal;
  std::shared_ptr<GlobalTensor> moleDeadlinesGlobal;
  std::shared_ptr<GlobalTensor> scoreGlobal;
  std::shared_ptr<GlobalTensor> realtimeSinceStartupGlobal;
  std::shared_ptr<GlobalTensor> moleOnTimingsGlobal;
  std::shared_ptr<GlobalTensor> moleOffTimingsGlobal;
    /**
     * The glTF to be driven by the skeleton animation.
     **/
  std::shared_ptr<GlobalTensor> poseMarkerGltf;
  std::shared_ptr<GlobalTensor> scoreboardGltf;
  std::array<std::shared_ptr<GlobalTensor>, MAX_MOLES> duckGltfs;
  std::array<std::shared_ptr<GlobalTensor>, MAX_JOINTS> redboxGltfs;
  std::shared_ptr<GlobalTensor> blankTextureGlobal;

  // Pipelines, as computation graphs able to be individually
  // scheduled, consisting of operators as nodes and local tensors
  // as edges.

  /**
   * The VST pipeline, for camera access
   */
  std::shared_ptr<Pipeline> m_secureMrVSTImagePipeline;
  /**
   * The pipeline where the detection algorithm is running. It intakes RGB image,
   * determine the region which it believes contains a human body, and outputs a
   * confidence score and an affine matrix from the RGB image to the region, in
   * <code>roiAffineGlobal</code>
   */
  std::shared_ptr<Pipeline> m_secureMrDetectionPipeline;
  /**
   * The pipeline to update the affine matrix in <code>roiAffineUpdatedGlobal</code>
   * using <code>roiAffineGlobal</code> <i>only</i> if the detection score is high;
   * otherwise, the old values in <code>roiAffineUpdatedGlobal</code> is retained.
   */
  std::shared_ptr<Pipeline> m_secureMrAffineUpdatePipeline;
  /**
   * The pipeline where the pose-landmark model is running. It applies the
   * affine matrix to get the image patch most likely containing the human body,
   * runs the pose-landmark models and compute each-bone transforms for skeleton
   * animation
   */
  std::shared_ptr<Pipeline> m_secureMrLandmarkPipeline;

  /**
   * Render pipeline, where the animation is updated timely
   */
  std::shared_ptr<Pipeline> m_secureMrRenderingPipeline;
  std::shared_ptr<Pipeline> m_secureMrMovePipeline;
  std::shared_ptr<Pipeline> m_secureMrInitMolesPipeline;

  // Placeholders for each pipeline
  // Recall placeholders are pipeline's local references to
  // global tensors, to avoid memory copy and competition
  // on the shared data between pipelines executed in different
  // threads.

  // Placeholders for the VST pipeline

  std::shared_ptr<PipelineTensor> vstOutputLeftUint8Placeholder;
  std::shared_ptr<PipelineTensor> vstOutputLeftFp32Placeholder;

  // Placeholders for the inference pipeline

  std::shared_ptr<PipelineTensor> smallF32ImagePlaceholder;
  std::shared_ptr<PipelineTensor> largeU8ImagePlaceholder;
  std::shared_ptr<PipelineTensor> isPoseDetectedPlaceholder;
  std::shared_ptr<PipelineTensor> bodyLandmarkPlaceholder;
  std::shared_ptr<PipelineTensor> molesTransformsPlaceholder;
  std::shared_ptr<PipelineTensor> moleCollisionsPlaceholder;
  std::shared_ptr<PipelineTensor> molesVisiblePlaceholder;
  std::shared_ptr<PipelineTensor> moleDeadlinesPlaceholder;
  std::shared_ptr<PipelineTensor> scorePlaceholder;
  std::shared_ptr<PipelineTensor> realtimeSinceStartupPlaceholder;
  std::shared_ptr<PipelineTensor> moleOnTimingsPlaceholder;
  std::shared_ptr<PipelineTensor> moleOffTimingsPlaceholder;
  /**
   * <code>roiAffinePh1</code> and <code>roiAffinePh2</code> refers to the same
   * <code>roiAffineGlobal</code>, from <code>m_secureMrDetectionPipeline</code>
   * and <code>m_secureMrAffineUpdatePipeline</code> respectively.
   * <br/>
   * <code>roiAffinePh3</code> and <code>roiAffinePh4</code> refers to the same
   * <code>roiAffineUpdatedGlobal</code>, from
   * <code>m_secureMrAffineUpdatePipeline</code>
   * and <code>m_secureMrLandmarkPipeline</code> respectively.
   */
  std::shared_ptr<PipelineTensor> roiAffinePh1, roiAffinePh2, roiAffinePh3, roiAffinePh4, roiImagePh;

  // PiPlaceholders for the render pipeline

  std::shared_ptr<PipelineTensor> gltfPlaceholderTensor;
  std::shared_ptr<PipelineTensor> gltfScoreboardPlaceholder;
  std::shared_ptr<PipelineTensor> isPoseDetectedPlaceholder2;
  std::shared_ptr<PipelineTensor> bodyLandmarkPlaceholder2;
  std::shared_ptr<PipelineTensor> molesTransformsPlaceholder2;
  std::shared_ptr<PipelineTensor> molesVisiblePlaceholder2;
  std::shared_ptr<PipelineTensor> currentScorePh;
  std::array<std::shared_ptr<PipelineTensor>, MAX_MOLES> moleGltfPlaceholders;
  std::array<std::shared_ptr<PipelineTensor>, MAX_MOLES> molePoseTensors;
  std::array<std::shared_ptr<PipelineTensor>, MAX_MOLES> moleVisibleTensors;

  std::array<std::shared_ptr<PipelineTensor>, MAX_JOINTS> redboxGltfPlaceholders;
  std::array<std::shared_ptr<PipelineTensor>, MAX_JOINTS> redboxPoseTensors;

    std::chrono::steady_clock::time_point startTime;

  // Pipeline placeholder/local tensor for the move pipeline
  std::shared_ptr<PipelineTensor> gltfPlaceholderTensor2;
  std::shared_ptr<PipelineTensor> stagePose;
  float stagePoseData[16]{
      1.0f, 0.0f, 0.0f, 0.6f,  //
      0.0f, 1.0f, 0.0f, -0.2f,  //
      0.0f, 0.0f, 1.0f, -2.5f,  //
      0.0f, 0.0f, 0.0f, 1.0f    //
  };
  std::shared_ptr<PipelineTensor> bodyLandmarkInitPlaceholder;
  std::shared_ptr<PipelineTensor> molesTransformsInitPlaceholder;
  std::shared_ptr<PipelineTensor> molesCollisionsInitPlaceholder;
  bool molesInitApplied = false;

  // Run-time control

  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;
  std::atomic<bool> keepRunning{true};
  std::atomic<bool> pipelineAllInitialized{false};
  std::vector<std::function<void()>> renderingInitializers;
};

}  // namespace SecureMR
