#pragma once

#include "pch.h"
#include <array>
#include <atomic>
#include <fstream>
#include <random>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <openxr/openxr.h>
#include <xr_linear.h>
#include <logger.h>
#include <common.h>
#include <securemr_base.h>
#include <securemr_utils/session.h>
#include <securemr_utils/pipeline.h>
#include <securemr_utils/tensor.h>
#include <securemr_utils/rendercommand.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// Forward declarations for OpenXR types
struct XrVector3f;

namespace SecureMR {

namespace Side {
constexpr size_t Left = 0;
constexpr size_t Right = 1;
constexpr size_t Count = 2;
}

// Forward declarations for SecureMR types
class ISecureMR;
class FrameworkSession;
class Pipeline;
class PipelineTensor;
class GlobalTensor;

class StylizeDemo final : public ISecureMR {
public:
  static constexpr const char* STYLE_PREDICTOR_MODEL_PATH = "stylepredict.tflite";
  static constexpr const char* STYLE_TRANSFER_MODEL_PATH = "styletransfer.tflite";
  static constexpr const char* PORTAL_GLTF_PATH = "portal.gltf";
  static constexpr const char* STYLE_TEXTURE_PATH1 = "textures/vangogh.png";
  static constexpr const char* STYLE_TEXTURE_PATH2 = "textures/kandinsky.png";
  static constexpr const char* STYLE_TEXTURE_PATH3 = "textures/altruria.jpg";
  static constexpr const char* STYLE_TEXTURE_PATH4 = "textures/geometricum.jpg";
  static constexpr const char* STYLE_TEXTURE_PATH5 = "textures/blechermosaic.jpg";

  static constexpr const char* STYLE_TEXTURE_PATHS[] = {
      STYLE_TEXTURE_PATH1,
      STYLE_TEXTURE_PATH2,
      STYLE_TEXTURE_PATH3,
      STYLE_TEXTURE_PATH4,
      STYLE_TEXTURE_PATH5
  };

  static constexpr int NUM_STYLE_TEXTURES = 5;

  explicit StylizeDemo(const XrInstance& instance, const XrSession& session);
  ~StylizeDemo() override;

  void CreateFramework() override;
  void CreatePipelines() override;
  void RunPipelines() override;
  void UpdateHandPose(const XrVector3f* leftHandDelta, const XrVector3f* rightHandDelta) override;
  void UpdateControllerPose(const XrPosef* leftPose, const XrPosef* rightPose, const XrView* views, uint32_t viewCount) override;
  void HandleButtonPress(int side = -1) override;
  [[nodiscard]]  bool WantsControllerVisualization() const override { return false; }
  void UpdateFrame();
  bool LoadingFinished() const override { return pipelineAllInitialized.load(); }
protected:
  void CreateGlobalTensor();
  void CreateSecureMrVSTImagePipeline();
  void CreateSecureMrStylePredictionPipeline();
  void CreateSecureMrStyleTransferPipeline();
  void CreateSecureMrRenderingPipeline();
  void LoadStyleTextureFromAssetPath(const char* styleTexturePath, std::shared_ptr<GlobalTensor> targetTensor);

  void UpdatePortalPose(const XrPosef& controllerPose, const XrView& view, std::shared_ptr<GlobalTensor> outputTensor, std::shared_ptr<GlobalTensor> srcPointsTensor, float xOffset);

  XrSecureMrPipelineRunPICO RunSecureMrVSTImagePipeline(XrSecureMrPipelineRunPICO pre) const;
  XrSecureMrPipelineRunPICO RunSecureMrStylePredictionPipeline(XrSecureMrPipelineRunPICO pre) const;
  XrSecureMrPipelineRunPICO RunSecureMrStyleTransferPipeline(XrSecureMrPipelineRunPICO pre) const;
  XrSecureMrPipelineRunPICO RunSecureMrRenderingPipeline(XrSecureMrPipelineRunPICO pre) const;

 private:
  XrInstance xr_instance{XR_NULL_HANDLE};
  XrSession xr_session{XR_NULL_HANDLE};
  std::shared_ptr<FrameworkSession> frameworkSession;

  // Global tensors
  std::shared_ptr<GlobalTensor> vstOutputLeftUint8Global;
  std::array<std::shared_ptr<GlobalTensor>, Side::Count> styleTextureGlobal;
  std::array<std::shared_ptr<GlobalTensor>, Side::Count> predictedStyleGlobal;
  std::array<std::shared_ptr<GlobalTensor>, Side::Count> portalGltf;
  std::array<std::shared_ptr<GlobalTensor>, Side::Count> portalPoseGlobal;
  std::array<std::shared_ptr<GlobalTensor>, Side::Count> srcPointsGlobal;

  std::array<std::shared_ptr<GlobalTensor>, Side::Count> stylizedImageGPUGlobal;

  // Pipelines
  std::shared_ptr<Pipeline> m_secureMrVSTImagePipeline;
  std::shared_ptr<Pipeline> m_secureMrStylePredictionPipeline;
  std::shared_ptr<Pipeline> m_secureMrStyleTransferPipeline;
  std::shared_ptr<Pipeline> m_secureMrRenderingPipeline;

  // Pipeline placeholders
  std::shared_ptr<PipelineTensor> vstOutputLeftUint8VstPlaceholder;
  std::shared_ptr<PipelineTensor> vstOutputLeftUint8StylePlaceholder;
  std::shared_ptr<PipelineTensor> styleTexturePlaceholder;
  std::shared_ptr<PipelineTensor> predictedStyleOutputPlaceholder;

  std::shared_ptr<PipelineTensor> predictedStyleInputPlaceholder;

  std::array<std::shared_ptr<PipelineTensor>, Side::Count> portalGltfPlaceholder;
  std::array<std::shared_ptr<PipelineTensor>, Side::Count> portalPosePlaceholder;
  std::shared_ptr<PipelineTensor> srcPointsPlaceholder;

  std::shared_ptr<PipelineTensor> predictedStyleInputPlaceholderGPU;
  std::shared_ptr<PipelineTensor> stylizedImageOutputPlaceholderGPU;


  // Runtime control
  std::vector<std::thread> pipelineRunners;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::condition_variable initialized;
  std::mutex initialized_mtx;
  std::mutex secureMrApiMutex;
  std::atomic<bool> keepRunning{true};
  std::atomic<bool> pipelineAllInitialized{false};
  bool newTextureChosen{true};
  std::atomic<bool> globalTensorsInitialized{false};
  std::array<int, Side::Count> styleTextureIndex{0, 1};

  StylizeDemo(const StylizeDemo&) = delete;
  StylizeDemo& operator=(const StylizeDemo&) = delete;
  StylizeDemo(StylizeDemo&&) = delete;
  StylizeDemo& operator=(StylizeDemo&&) = delete;
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(XrInstance instance, XrSession session);

} // namespace SecureMR
