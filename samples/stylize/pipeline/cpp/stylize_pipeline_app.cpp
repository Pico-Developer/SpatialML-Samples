#include "pch.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include "securemr_utils/tensor.h"
#include "securemr_utils/utils.h"
#include "xr_linear.h"

#define STB_IMAGE_IMPLEMENTATION
#include "image_utils/stb_image.h"

namespace SecureMR {
namespace {

constexpr size_t kLeft = 0;
constexpr size_t kRight = 1;
constexpr size_t kSideCount = 2;
constexpr int kStyleSize = 256;
constexpr int kFrameWidth = 2048;
constexpr int kFrameHeight = 1536;
constexpr std::array<const char*, 5> kStylePaths{
    "textures/vangogh.png", "textures/kandinsky.png",
    "textures/altruria.jpg", "textures/geometricum.jpg",
    "textures/blechermosaic.jpg"};

std::shared_ptr<GlobalTensor> FindGlobal(const ModelPackagePipelineBundle& bundle,
                                         const char* name) {
  const auto it = bundle.globalTensorMap.find(name);
  return it == bundle.globalTensorMap.end() ? nullptr : it->second;
}

std::shared_ptr<GlobalTensor> CopyGlobal(const ModelPackagePipelineBundle& bundle,
                                         const char* name) {
  const auto source = FindGlobal(bundle, name);
  return source ? std::make_shared<GlobalTensor>(*source) : nullptr;
}

bool LoadStyle(const char* path, std::vector<uint8_t>& pixels) {
  std::vector<char> encoded;
  std::string error;
  if (!SecureMrUtils::LoadAssetToBuffer(path, encoded, &error)) {
    Log::Write(Log::Level::Error, Fmt("Stylize Pipeline Loader: %s", error.c_str()));
    return false;
  }
  int width = 0;
  int height = 0;
  int channels = 0;
  auto* decoded = stbi_load_from_memory(
      reinterpret_cast<const unsigned char*>(encoded.data()),
      static_cast<int>(encoded.size()), &width, &height, &channels, 3);
  if (!decoded || width != kStyleSize || height != kStyleSize) {
    if (decoded) stbi_image_free(decoded);
    Log::Write(Log::Level::Error,
               Fmt("Stylize Pipeline Loader: invalid style %s (%dx%d)", path, width, height));
    return false;
  }
  pixels.assign(decoded, decoded + width * height * 3);
  stbi_image_free(decoded);
  return true;
}

}  // namespace

class StylizePipelineApp final : public ISecureMR {
 public:
  StylizePipelineApp(const XrInstance& instance, const XrSession& session)
      : instance_(instance), session_(session) {}

  ~StylizePipelineApp() override {
    keepRunning_ = false;
    initialized_.notify_all();
    if (initializer_ && initializer_->joinable()) initializer_->join();
    keepRunning_ = false;
    if (runner_.joinable()) runner_.join();
  }

  void CreateFramework() override {
    framework_ = std::make_shared<FrameworkSession>(instance_, session_, kFrameWidth, kFrameHeight);
  }

  void CreatePipelines() override {
    initializer_ = std::make_unique<std::thread>([this]() {
      std::string error;
      bool loaded = SecureMrUtils::LoadModelPackagePipelinesFromAssets(
          "stylize-pipeline", framework_, {}, bundle_, error);
      if (loaded) loaded = InitializeSides(error);
      if (!loaded) {
        Log::Write(Log::Level::Error,
                   Fmt("Stylize Pipeline Loader: initialization failed: %s", error.c_str()));
      } else {
        ready_ = true;
        Log::Write(Log::Level::Info, "Stylize Pipeline Loader: package loaded");
      }
      keepRunning_ = loaded;
      initializedFlag_ = true;
      initialized_.notify_all();
    });
  }

  void RunPipelines() override {
    runner_ = std::thread([this]() {
      {
        std::unique_lock<std::mutex> lock(initializedMutex_);
        initialized_.wait(lock, [this]() {
          return initializedFlag_.load() || !keepRunning_.load();
        });
      }
      while (keepRunning_.load()) {
        {
          std::lock_guard<std::mutex> lock(apiMutex_);
          SubmitSide(kLeft);
          SubmitSide(kRight);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  void UpdateControllerPose(const XrPosef* leftPose, const XrPosef* rightPose,
                            const XrView* views, uint32_t viewCount) override {
    if (!ready_.load() || views == nullptr || viewCount == 0) return;
    std::lock_guard<std::mutex> lock(apiMutex_);
    if (leftPose) UpdateCirclePoses(kLeft, *leftPose, views[0], 0.2f);
    if (rightPose) UpdateCirclePoses(kRight, *rightPose, views[0], -0.8f);
  }

  void HandleButtonPress(int side) override {
    if (!ready_.load()) return;
    std::lock_guard<std::mutex> lock(apiMutex_);
    if (side == static_cast<int>(kLeft) || side == -1) AdvanceStyle(kLeft);
    if (side == static_cast<int>(kRight) || side == -1) AdvanceStyle(kRight);
  }

  [[nodiscard]] bool LoadingFinished() const override { return initializedFlag_.load(); }
  [[nodiscard]] bool WantsControllerVisualization() const override { return false; }

 private:
  bool InitializeSides(std::string& error) {
    const auto styleIt = bundle_.pipelines.find("style");
    const auto displayIt = bundle_.pipelines.find("display");
    if (styleIt == bundle_.pipelines.end() || displayIt == bundle_.pipelines.end()) {
      error = "style or display pipeline missing";
      return false;
    }
    style_ = &styleIt->second;
    display_ = &displayIt->second;

    std::vector<char> portalBytes;
    if (!SecureMrUtils::LoadAssetToBuffer("stylize-pipeline/gltf/portal.gltf",
                                          portalBytes, &error)) return false;

    for (size_t side = 0; side < kSideCount; ++side) {
      // Both sides need independent globals. Reusing the loader's first shared
      // bindings left side 0 at its initialization pose while side 1 moved.
      styleImage_[side] = CopyGlobal(bundle_, "style_image");
      srcPoints_[side] = CopyGlobal(bundle_, "src_points");
      stylizedImage_[side] = CopyGlobal(bundle_, "stylized_image");
      portalPose_[side] = CopyGlobal(bundle_, "portal_pose");
      portal_[side] =
          std::make_shared<GlobalTensor>(framework_, portalBytes.data(), portalBytes.size());
      if (!styleImage_[side] || !srcPoints_[side] || !stylizedImage_[side] ||
          !portalPose_[side] || !portal_[side]) {
        error = Fmt("missing loader binding for side %zu", side);
        return false;
      }
      if (!SetStyle(side, styleIndex_[side])) {
        error = Fmt("failed to load style for side %zu", side);
        return false;
      }
      const float portalMatrix[16] = {
          1,0,0,side == kLeft ? -0.65f : 0.65f, 0,1,0,0.25f, 0,0,1,-2.0f, 0,0,0,1};
      portalPose_[side]->setData(
          reinterpret_cast<int8_t*>(const_cast<float*>(portalMatrix)), sizeof(portalMatrix));
      const float points[6] = {side == kLeft ? 256.0f : 1024.0f, 256.0f,
                               side == kLeft ? 1024.0f : 1792.0f, 256.0f,
                               side == kLeft ? 256.0f : 1024.0f, 1024.0f};
      srcPoints_[side]->setData(reinterpret_cast<int8_t*>(const_cast<float*>(points)), sizeof(points));
    }
    return true;
  }

  std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> Bindings(
      ModelPackagePipeline& package,
      const std::unordered_map<std::string, std::shared_ptr<GlobalTensor>>& overrides) {
    auto bindings = package.submitBindings;
    for (const auto& [name, global] : overrides) {
      const auto it = package.tensorMap.find(name);
      if (it != package.tensorMap.end()) bindings[it->second] = global;
    }
    return bindings;
  }

  void SubmitSide(size_t side) {
    const auto styleRun = style_->pipeline->submit(
        Bindings(*style_, {{"style_image", styleImage_[side]},
                           {"src_points", srcPoints_[side]},
                           {"stylized_image", stylizedImage_[side]}}),
        XR_NULL_HANDLE, nullptr);
    display_->pipeline->submit(
        Bindings(*display_, {{"stylized_image", stylizedImage_[side]},
                             {"portal", portal_[side]},
                             {"portal_pose", portalPose_[side]}}),
        styleRun, nullptr);
  }

  bool SetStyle(size_t side, size_t index) {
    std::vector<uint8_t> pixels;
    if (!LoadStyle(kStylePaths[index], pixels)) return false;
    styleImage_[side]->setData(reinterpret_cast<int8_t*>(pixels.data()), pixels.size());
    Log::Write(Log::Level::Info,
               Fmt("Stylize Pipeline Loader: %s controller selected %s",
                   side == kLeft ? "left" : "right", kStylePaths[index]));
    return true;
  }

  void AdvanceStyle(size_t side) {
    const size_t next = (styleIndex_[side] + 1) % kStylePaths.size();
    if (SetStyle(side, next)) styleIndex_[side] = next;
  }

  void UpdateCirclePoses(size_t side, const XrPosef& controllerPose,
                         const XrView& view, float xOffset) {
    XrMatrix4x4f worldFromView;
    XrMatrix4x4f_CreateFromRigidTransform(&worldFromView, &view.pose);
    XrMatrix4x4f viewFromWorld;
    XrMatrix4x4f_InvertRigidBody(&viewFromWorld, &worldFromView);
    XrVector3f controllerView;
    XrMatrix4x4f_TransformVector3f(&controllerView, &viewFromWorld, &controllerPose.position);

    const float portalT = std::fabs(controllerView.z) > 1e-5f ? -2.0f / controllerView.z : 1.0f;
    const float portalMatrix[16] = {
        1,0,0,controllerView.x * portalT + xOffset,
        0,1,0,controllerView.y * portalT + 0.8f,
        0,0,1,controllerView.z * portalT, 0,0,0,1};
    portalPose_[side]->setData(
        reinterpret_cast<int8_t*>(const_cast<float*>(portalMatrix)), sizeof(portalMatrix));

    const XrVector3f corners[3] = {
        {-0.51f, 0.68f, 0}, {0.51f, 0.68f, 0}, {-0.51f, -0.68f, 0}};
    XrMatrix4x4f portal;
    std::memcpy(portal.m, portalMatrix, sizeof(portalMatrix));
    XrMatrix4x4f modelView;
    XrMatrix4x4f_Transpose(&modelView, &portal);
    XrMatrix4x4f projection;
    XrMatrix4x4f_CreateProjectionFov(
        &projection, GRAPHICS_OPENGL_ES, view.fov, 0.1f, 100.0f);
    XrMatrix4x4f mvp;
    XrMatrix4x4f_Multiply(&mvp, &projection, &modelView);
    std::array<XrVector2f, 3> screen;
    for (size_t i = 0; i < screen.size(); ++i) {
      XrVector3f ndc;
      XrMatrix4x4f_TransformVector3f(&ndc, &mvp, &corners[i]);
      screen[i] = {(ndc.x + 1.0f) * 0.5f * kFrameWidth,
                   (1.0f - ndc.y) * 0.5f * kFrameHeight};
    }
    srcPoints_[side]->setData(reinterpret_cast<int8_t*>(screen.data()), sizeof(screen));
  }

  XrInstance instance_{XR_NULL_HANDLE};
  XrSession session_{XR_NULL_HANDLE};
  std::shared_ptr<FrameworkSession> framework_;
  ModelPackagePipelineBundle bundle_;
  ModelPackagePipeline* style_{nullptr};
  ModelPackagePipeline* display_{nullptr};
  std::array<std::shared_ptr<GlobalTensor>, kSideCount> styleImage_;
  std::array<std::shared_ptr<GlobalTensor>, kSideCount> srcPoints_;
  std::array<std::shared_ptr<GlobalTensor>, kSideCount> stylizedImage_;
  std::array<std::shared_ptr<GlobalTensor>, kSideCount> portal_;
  std::array<std::shared_ptr<GlobalTensor>, kSideCount> portalPose_;
  std::array<size_t, kSideCount> styleIndex_{0, 1};
  std::unique_ptr<std::thread> initializer_;
  std::thread runner_;
  std::condition_variable initialized_;
  std::mutex initializedMutex_;
  std::mutex apiMutex_;
  std::atomic<bool> keepRunning_{true};
  std::atomic<bool> initializedFlag_{false};
  std::atomic<bool> ready_{false};
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance,
                                                 const XrSession& session) {
  return std::make_shared<StylizePipelineApp>(instance, session);
}

}  // namespace SecureMR
