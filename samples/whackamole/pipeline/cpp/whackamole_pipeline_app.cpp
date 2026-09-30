#include "pch.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "logger.h"
#include "securemr_base.h"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/session.h"
#include "securemr_utils/utils.h"

namespace SecureMR {

namespace {

constexpr char kPackageRoot[] = "whackamole-pipeline";
constexpr char kTimeTensor[] = "realtime_since_startup";
constexpr char kScoreboardInitPipeline[] = "scoreboard_init";
constexpr char kInferencePipeline[] = "inference";
constexpr char kDisplayPipeline[] = "display";

}  // namespace

class WhackamolePipelineApp final : public ISecureMR {
 public:
  WhackamolePipelineApp(const XrInstance& instance, const XrSession& session)
      : instance_(instance), session_(session) {}

  ~WhackamolePipelineApp() override {
    keepRunning_ = false;
    initialized_.notify_all();
    if (initializer_ && initializer_->joinable()) initializer_->join();
    if (runner_.joinable()) runner_.join();
  }

  void CreateFramework() override {
    framework_ = std::make_shared<FrameworkSession>(instance_, session_, 512, 512);
  }

  void CreatePipelines() override {
    initializer_ = std::make_unique<std::thread>([this]() {
      timeGlobal_ = std::make_shared<GlobalTensor>(
          framework_, TensorAttribute_ScalarArray{1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO});
      *timeGlobal_ = std::vector<float>{0.0f};

      std::string error;
      bool loaded = false;
      constexpr int kPackageLoadAttempts = 3;
      for (int attempt = 1; attempt <= kPackageLoadAttempts && !loaded; ++attempt) {
        loaded = SecureMrUtils::LoadModelPackagePipelinesFromAssets(
            kPackageRoot, framework_, {{kTimeTensor, timeGlobal_}}, bundle_, error);
        if (!loaded && attempt < kPackageLoadAttempts) {
          Log::Write(Log::Level::Warning,
                     Fmt("Whack-a-Mole Pipeline: package load attempt %d failed: %s; retrying",
                         attempt, error.c_str()));
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
      }
      if (!loaded) {
        Log::Write(Log::Level::Error,
                   Fmt("Whack-a-Mole Pipeline: package load failed: %s", error.c_str()));
      }

      scoreboardInit_ = FindPipeline(kScoreboardInitPipeline);
      inference_ = FindPipeline(kInferencePipeline);
      display_ = FindPipeline(kDisplayPipeline);
      keepRunning_ = loaded && scoreboardInit_ != nullptr && inference_ != nullptr && display_ != nullptr;
      if (keepRunning_) {
        timerPipeline_ = std::make_shared<Pipeline>(framework_);
        timerOutput_ = PipelineTensor::PipelinePlaceholderLike(timerPipeline_, timeGlobal_);
        const auto startMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string script =
            "var elapsed; elapsed[0] = (Date.now() - " + std::to_string(startMs) + ") / 1000.0;";
        timerPipeline_->runJavascript(script.data(), script.size(), {}, {{"elapsed", timerOutput_}});
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        initializedFlag_ = true;
      }
      initialized_.notify_all();
    });
  }

  void RunPipelines() override {
    runner_ = std::thread([this]() {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        initialized_.wait(lock, [this]() { return initializedFlag_ || !keepRunning_; });
      }
      if (!keepRunning_) return;

      XrSecureMrPipelineRunPICO initializationRun =
          scoreboardInit_->pipeline->submit(scoreboardInit_->submitBindings, XR_NULL_HANDLE, nullptr);
      auto previousRun = initializationRun;
      while (keepRunning_) {
        const auto timerRun = timerPipeline_->submit({{timerOutput_, timeGlobal_}}, previousRun, nullptr);

        const XrSecureMrPipelineRunPICO inferenceRun = inference_->pipeline->submit(
            inference_->submitBindings, timerRun, nullptr);
        previousRun = display_->pipeline->submit(display_->submitBindings, inferenceRun, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  [[nodiscard]] bool LoadingFinished() const override { return initializedFlag_; }

 private:
  ModelPackagePipeline* FindPipeline(const std::string& id) {
    const auto it = bundle_.pipelines.find(id);
    if (it == bundle_.pipelines.end() || it->second.pipeline == nullptr) {
      Log::Write(Log::Level::Error,
                 Fmt("Whack-a-Mole Pipeline: package pipeline '%s' is missing", id.c_str()));
      return nullptr;
    }
    return &it->second;
  }

  XrInstance instance_;
  XrSession session_;
  std::shared_ptr<FrameworkSession> framework_;
  std::shared_ptr<GlobalTensor> timeGlobal_;
  ModelPackagePipelineBundle bundle_;
  ModelPackagePipeline* scoreboardInit_ = nullptr;
  ModelPackagePipeline* inference_ = nullptr;
  ModelPackagePipeline* display_ = nullptr;
  std::unique_ptr<std::thread> initializer_;
  std::thread runner_;
  std::condition_variable initialized_;
  std::mutex mutex_;
  std::shared_ptr<Pipeline> timerPipeline_;
  std::shared_ptr<PipelineTensor> timerOutput_;
  std::atomic<bool> keepRunning_{true};
  std::atomic<bool> initializedFlag_{false};
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance,
                                                 const XrSession& session) {
  return std::make_shared<WhackamolePipelineApp>(instance, session);
}

}  // namespace SecureMR
