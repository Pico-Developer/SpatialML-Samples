#include "pch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "logger.h"
#include "securemr_base.h"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/session.h"
#include "securemr_utils/utils.h"

#ifndef PIPELINE_PACKAGE_ROOT
#error "PIPELINE_PACKAGE_ROOT must be defined"
#endif
#ifndef PIPELINE_IDS
#error "PIPELINE_IDS must be defined"
#endif
#ifndef PIPELINE_APP_TAG
#define PIPELINE_APP_TAG "PipelineSample"
#endif
#ifndef PIPELINE_FRAME_WIDTH
#define PIPELINE_FRAME_WIDTH 512
#endif
#ifndef PIPELINE_FRAME_HEIGHT
#define PIPELINE_FRAME_HEIGHT 512
#endif
#ifndef PIPELINE_SUBMIT_INTERVAL_MS
#define PIPELINE_SUBMIT_INTERVAL_MS 50
#endif
#ifndef PIPELINE_PERIODS
#define PIPELINE_PERIODS ""
#endif
#ifndef PIPELINE_INITIAL_BURST_FRAMES
#define PIPELINE_INITIAL_BURST_FRAMES 0
#endif
namespace {

std::vector<std::string> SplitList(const std::string& value) {
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) result.push_back(item);
  }
  return result;
}

std::vector<uint32_t> ParsePeriods(const std::string& value, const size_t count) {
  std::vector<uint32_t> result(count, 1u);
  if (value.empty()) return result;
  const auto entries = SplitList(value);
  for (size_t i = 0; i < entries.size() && i < result.size(); ++i) {
    try {
      result[i] = std::max(1, std::stoi(entries[i]));
    } catch (...) {
      result[i] = 1u;
    }
  }
  return result;
}

}  // namespace

namespace SecureMR {

class PipelinePackageApp final : public ISecureMR {
 public:
  PipelinePackageApp(const XrInstance& instance, const XrSession& session)
      : instance_(instance), session_(session) {}

  ~PipelinePackageApp() override {
    keepRunning_ = false;
    initialized_.notify_all();
    if (initializer_ && initializer_->joinable()) initializer_->join();
    for (auto& runner : runners_) {
      if (runner.joinable()) runner.join();
    }
  }

  void CreateFramework() override {
    framework_ = std::make_shared<FrameworkSession>(instance_, session_, PIPELINE_FRAME_WIDTH,
                                                    PIPELINE_FRAME_HEIGHT);
  }

  void CreatePipelines() override {
    initializer_ = std::make_unique<std::thread>([this]() {
      std::string error;
      const bool loaded = SecureMrUtils::LoadModelPackagePipelinesFromAssets(
          PIPELINE_PACKAGE_ROOT, framework_, {}, bundle_, error);
      if (!loaded) {
        Log::Write(Log::Level::Error,
                   Fmt("%s: package load failed: %s", PIPELINE_APP_TAG, error.c_str()));
      }

      bool ready = loaded;
      const auto pipelineIds = SplitList(PIPELINE_IDS);
      periods_ = ParsePeriods(PIPELINE_PERIODS, pipelineIds.size());
      for (const auto& id : pipelineIds) {
        const auto it = bundle_.pipelines.find(id);
        if (it == bundle_.pipelines.end() || it->second.pipeline == nullptr) {
          Log::Write(Log::Level::Error,
                     Fmt("%s: package pipeline '%s' is missing", PIPELINE_APP_TAG, id.c_str()));
          ready = false;
        } else {
          orderedPipelines_.push_back(&it->second);
        }
      }

      keepRunning_ = ready;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        initializedFlag_ = true;
      }
      initialized_.notify_all();
    });
  }

  void RunPipelines() override {
    runners_.emplace_back([this]() {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        initialized_.wait(lock, [this]() { return initializedFlag_ || !keepRunning_; });
      }
      uint64_t frameIndex = 0;
      while (keepRunning_) {
        XrSecureMrPipelineRunPICO previous = XR_NULL_HANDLE;
        for (size_t i = 0; i < orderedPipelines_.size(); ++i) {
          const bool initialBurst = frameIndex < PIPELINE_INITIAL_BURST_FRAMES;
          if (!initialBurst && frameIndex % periods_[i] != 0) continue;
          const auto* package = orderedPipelines_[i];
          previous = package->pipeline->submit(package->submitBindings, previous, nullptr);
        }
        ++frameIndex;
        std::this_thread::sleep_for(std::chrono::milliseconds(PIPELINE_SUBMIT_INTERVAL_MS));
      }
    });
  }

  [[nodiscard]] bool LoadingFinished() const override { return initializedFlag_; }

 private:
  XrInstance instance_;
  XrSession session_;
  std::shared_ptr<FrameworkSession> framework_;
  ModelPackagePipelineBundle bundle_;
  std::vector<const ModelPackagePipeline*> orderedPipelines_;
  std::vector<uint32_t> periods_;
  std::unique_ptr<std::thread> initializer_;
  std::vector<std::thread> runners_;
  std::condition_variable initialized_;
  std::mutex mutex_;
  std::atomic<bool> keepRunning_{true};
  std::atomic<bool> initializedFlag_{false};
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance,
                                                 const XrSession& session) {
  return std::make_shared<PipelinePackageApp>(instance, session);
}

}  // namespace SecureMR
