#pragma once

#include "pch.h"
#include <atomic>
#include <array>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "logger.h"
#include "common.h"
#include "securemr_base.h"
#include "securemr_utils/adapter.hpp"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/rendercommand.h"
#include "securemr_utils/session.h"
#include "securemr_utils/tensor.h"

namespace SecureMR {

class MnistWildApp : public ISecureMR {
 public:
  static constexpr int kImageWidth = 3248;
  static constexpr int kImageHeight = 2464;
  static constexpr int kCropWidth = 224;
  static constexpr int kCropHeight = 224;

  MnistWildApp(const XrInstance& instance, const XrSession& session);
  ~MnistWildApp() override;

  void CreateFramework() override;
  void CreatePipelines() override;
  void RunPipelines() override;
  void RequestPermission(struct android_app* app) override;
  [[nodiscard]] bool LoadingFinished() const override { return pipelinesReady; }

 private:
  void CreateGlobalTensors();
  void CreateInferencePipeline();
  void CreateRenderPipeline();
  void RunInferencePipeline();
  void RunRenderPipeline();
  bool LoadAsset(const std::string& filePath, std::vector<char>& data) const;
  bool DeserializeInferencePipeline(const std::filesystem::path& jsonPath);

  XrInstance xr_instance;
  XrSession xr_session;

  std::shared_ptr<FrameworkSession> frameworkSession;
  std::shared_ptr<GlobalTensor> predictedClassGlobal;
  std::shared_ptr<GlobalTensor> predictedScoreGlobal;
  std::shared_ptr<GlobalTensor> croppedImageGlobal;
  std::shared_ptr<GlobalTensor> gltfClassAsset;
  std::shared_ptr<GlobalTensor> gltfScoreAsset;
  std::shared_ptr<GlobalTensor> gltfImageAsset;

  std::shared_ptr<Pipeline> inferencePipeline;
  std::shared_ptr<Pipeline> renderPipeline;

  std::shared_ptr<PipelineTensor> predClassPlaceholder;
  std::shared_ptr<PipelineTensor> predScorePlaceholder;
  std::shared_ptr<PipelineTensor> cropImagePlaceholder;

  std::shared_ptr<PipelineTensor> renderClassPlaceholder;
  std::shared_ptr<PipelineTensor> renderScorePlaceholder;
  std::shared_ptr<PipelineTensor> renderCropPlaceholder;
  std::shared_ptr<PipelineTensor> renderClassGltfPlaceholder;
  std::shared_ptr<PipelineTensor> renderScoreGltfPlaceholder;
  std::shared_ptr<PipelineTensor> renderImageGltfPlaceholder;
  std::unique_ptr<std::thread> pipelineInitializer;
  std::vector<std::thread> pipelineRunners;
  mutable std::mutex initMutex;
  std::condition_variable initCv;
  bool pipelinesReady = false;
  bool keepRunning = true;
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session);

}  // namespace SecureMR
