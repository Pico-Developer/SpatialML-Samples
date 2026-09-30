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
#include "securemr_utils/readback.h"

#define READBACK_MODEL_PATH "detection.tflite"
#define POSE_LANDMARK_MODEL_PATH "landmark.tflite"
#define GLTF_PATH "pose_marker.gltf"
#define ANCHOR_MAT "anchors_1.mat"
#include "stb_image.h"
#include "stb_image_write.h"

#ifdef __cplusplus
extern "C" {
#endif

static std::mutex g_permMutex;
static bool gPermissionCamera = false;

JNIEXPORT void JNICALL
Java_com_bytedance_pico_spatial_1ml_1demo_readback_ReadbackActivity_nativeSetPermission(
    JNIEnv* env,
    jclass,
    jstring permission,
    jboolean granted);
#ifdef __cplusplus
}
#endif

namespace SecureMR {

class ReadbackCheck : public ISecureMR {
 public:
  struct Config {
    int w = 512;
    int h = 512;
  };
  struct ImageRGB {
    int w = 0, h = 0;
    std::vector<uint8_t> rgb;
    bool valid() const { return w > 0 && h > 0 && rgb.size() == (size_t)w * h * 3; }
  };

  static struct android_app *gapp;

  ReadbackCheck(const XrInstance& instance, const XrSession& session);
  ~ReadbackCheck() override;
  void CreateFramework() override;
  void CreatePipelines() override;
  void RunPipelines() override;
  void RequestPermission(struct android_app* app) override;
  void Tick() override;

  [[nodiscard]] bool LoadingFinished() const override { return pipelineAllInitialized; }

 protected:
  /**
   * Create all the global tensors, must be called before create any pipelines
   */
  void CreateGlobalTensor();
  void CreateRelaxMrReadBackPipeline();

  XrSecureMrPipelineRunPICO RunRelaxMrReadBackPipeline(const XrSecureMrPipelineRunPICO pre = XR_NULL_HANDLE);

  void initializeGraphicsContext();
  void OutputReadbackBufferToFile(const XrReadbackTensorBufferPICO* tensorBuffer, const std::string &path);
  bool OutputReadbackTextureToPath(const XrReadbackTexturePICO& texture, const std::string& path);
#ifdef XR_USE_GRAPHICS_API_VULKAN
  VkQueue queue;
  VkCommandPool cmdPool;
  void OutputVulkanTextureToPath(XrReadbackTextureImageVulkanPICO * vTexture, const std::string &path);
#endif
#ifdef XR_USE_GRAPHICS_API_OPENGL_ES
  void OutputOpenGLTextureToPath(XrReadbackTextureImageOpenGLPICO * vTexture, const std::string &path);
#endif

  XrInstance xr_instance;
  XrSession xr_session;
  static bool isCpuBuffer;

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

  std::shared_ptr<GlobalTensor> vstOutputLeftEyeUint8Global;

  /**
   * The pipeline where the detection algorithm is running. It intakes RGB image,
   * determine the region which it believes contains a human body, and outputs a
   * confidence score and an affine matrix from the RGB image to the region, in
   * <code>roiAffineGlobal</code>
   */
  std::shared_ptr<Pipeline> m_RelaxMrReadBackPipeline;

  // Placeholders for each pipeline
  // Recall placeholders are pipeline's local references to
  // global tensors, to avoid memory copy and competition
  // on the shared data between pipelines executed in different
  // threads.

  // Placeholders for the VST pipeline

  std::shared_ptr<PipelineTensor> vstOutputLeftUint8Placeholder;
 
  // Run-time control

  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;

  bool keepRunning = true;
  bool pipelineAllInitialized = false;

  // controller for readback
  ReadbackController* mReadbackController = nullptr;
  ReadbackController::ReadbackRequest *mCurrentReadbackRequest = nullptr;
  XrReadbackTexturePICO readbackTexture = XR_NULL_HANDLE;
  Config mConfig;
};

}  // namespace SecureMR
