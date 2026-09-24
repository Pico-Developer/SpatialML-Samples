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

#include "pch.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/system_properties.h>

#include "common.h"
#include "logger.h"
#include "securemr_base.h"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/readback_async.h"
#include "securemr_utils/serialization.h"
#include "securemr_utils/session.h"
#include "securemr_utils/tensor.h"
#include "securemr_utils/utils.h"
#include "openxr_program.h"

extern std::string g_internalDataPath;

namespace SecureMR {
namespace {

constexpr char kModelDirProp[] = "debug.securemr.model_inspect.model_dir";
constexpr char kInputFileProp[] = "debug.securemr.model_inspect.input";
constexpr char kModelBinName[] = "model.serialized.bin";
constexpr char kModelJsonName[] = "model.serialized.json";

std::string ReadProp(const char* key) {
  char value[PROP_VALUE_MAX] = {};
  if (__system_property_get(key, value) == 0) {
    return {};
  }
  return std::string(value);
}

std::optional<std::filesystem::path> GetExternalFilesDir() {
  android_app* app = IOpenXrProgram::gapp;
  if (app == nullptr || app->activity == nullptr || app->activity->vm == nullptr || app->activity->clazz == nullptr) {
    Log::Write(Log::Level::Warning, "ModelInspect: android_app not ready; cannot query external files dir");
    return std::nullopt;
  }
  JNIEnv* env = nullptr;
  if (app->activity->vm->AttachCurrentThread(&env, nullptr) != JNI_OK || env == nullptr) {
    Log::Write(Log::Level::Warning, "ModelInspect: failed to attach JNI env; cannot query external files dir");
    return std::nullopt;
  }

  jclass activityCls = env->GetObjectClass(app->activity->clazz);
  if (activityCls == nullptr) {
    Log::Write(Log::Level::Warning, "ModelInspect: failed to get Activity class");
    return std::nullopt;
  }

  jmethodID midGetExternalFilesDir =
      env->GetMethodID(activityCls, "getExternalFilesDir", "(Ljava/lang/String;)Ljava/io/File;");
  if (midGetExternalFilesDir == nullptr) {
    Log::Write(Log::Level::Warning, "ModelInspect: getExternalFilesDir method not found");
    env->DeleteLocalRef(activityCls);
    return std::nullopt;
  }

  jobject fileObj = env->CallObjectMethod(app->activity->clazz, midGetExternalFilesDir, nullptr);
  if (fileObj == nullptr) {
    Log::Write(Log::Level::Warning, "ModelInspect: getExternalFilesDir returned null");
    env->DeleteLocalRef(activityCls);
    return std::nullopt;
  }

  jclass fileCls = env->GetObjectClass(fileObj);
  jmethodID midGetAbsolutePath = env->GetMethodID(fileCls, "getAbsolutePath", "()Ljava/lang/String;");
  if (midGetAbsolutePath == nullptr) {
    Log::Write(Log::Level::Warning, "ModelInspect: getAbsolutePath not found on File");
    env->DeleteLocalRef(fileCls);
    env->DeleteLocalRef(fileObj);
    env->DeleteLocalRef(activityCls);
    return std::nullopt;
  }

  jstring pathStr = static_cast<jstring>(env->CallObjectMethod(fileObj, midGetAbsolutePath));
  std::optional<std::filesystem::path> result;
  if (pathStr != nullptr) {
    const char* pathChars = env->GetStringUTFChars(pathStr, nullptr);
    if (pathChars != nullptr) {
      result = std::filesystem::path(pathChars);
      env->ReleaseStringUTFChars(pathStr, pathChars);
    }
    env->DeleteLocalRef(pathStr);
  }

  env->DeleteLocalRef(fileCls);
  env->DeleteLocalRef(fileObj);
  env->DeleteLocalRef(activityCls);
  if (!result.has_value()) {
    Log::Write(Log::Level::Warning, "ModelInspect: external files dir path unavailable");
  }
  return result;
}
}  // namespace

class ModelInspectApp : public ISecureMR {
 public:
  ModelInspectApp(const XrInstance& instance, const XrSession& session)
      : xrInstance_(instance), xrSession_(session) {}

  ~ModelInspectApp() override {
    if (readback_) {
      readback_->Stop();
    }
    if (runThread_ && runThread_->joinable()) {
      runThread_->join();
    }
    if (initThread_ && initThread_->joinable()) {
      initThread_->join();
    }
  }

  void CreateFramework() override {
    Log::Write(Log::Level::Info, "ModelInspect: creating framework session");
    frameworkSession_ = std::make_shared<FrameworkSession>(xrInstance_, xrSession_, 640, 480);
    // Prefer external app-specific storage so dumps are readable without root.
    const auto externalDir = GetExternalFilesDir();
    const std::filesystem::path preferredOutputDir =
        externalDir.has_value() ? *externalDir / "model_inspect" : std::filesystem::path();
    const std::filesystem::path fallbackOutputDir("/data/local/tmp/securemr_model_inspect");
    std::error_code ec;
    if (externalDir.has_value()) {
      outputDir_ = preferredOutputDir;
      std::filesystem::create_directories(outputDir_, ec);
      if (ec) {
        Log::Write(Log::Level::Warning,
                   Fmt("ModelInspect: failed to ensure output dir %s (%s), falling back to %s",
                       outputDir_.string().c_str(), ec.message().c_str(), fallbackOutputDir.string().c_str()));
      }
    }
    if (!externalDir.has_value() || ec) {
      outputDir_ = fallbackOutputDir;
      ec.clear();
      std::filesystem::create_directories(outputDir_, ec);
    }
    if (ec) {
      Log::Write(Log::Level::Warning,
                 Fmt("ModelInspect: failed to ensure fallback output dir %s (%s)",
                     outputDir_.string().c_str(), ec.message().c_str()));
    } else {
      Log::Write(Log::Level::Info, Fmt("ModelInspect: output dir %s", outputDir_.string().c_str()));
    }
  }

  void CreatePipelines() override {
    initThread_ = std::make_unique<std::thread>([this]() { Initialize(); });
  }

  void RunPipelines() override {
    runThread_ = std::make_unique<std::thread>([this]() {
      std::unique_lock<std::mutex> lock(initMutex_);
      initCv_.wait(lock, [this]() { return initFinished_; });
      if (!pipelineReady_ || !pipeline_) {
        Log::Write(Log::Level::Error, "ModelInspect: initialization failed, skip inference run");
        return;
      }
      lock.unlock();

      Log::Write(Log::Level::Info, "ModelInspect: starting readback");
      if (readback_) {
        readback_->Start();
      }

      LogPlaceholderMappings();
      Log::Write(Log::Level::Info, "ModelInspect: submitting inference pipeline");
      pipeline_->submit(placeholderMap_, XR_NULL_HANDLE, nullptr);

      {
        std::unique_lock<std::mutex> rbLock(readbackMutex_);
        if (!readbackCv_.wait_for(rbLock, std::chrono::seconds(5),
                                  [this]() { return readbackDone_.load(std::memory_order_acquire); })) {
          Log::Write(Log::Level::Warning, "ModelInspect: readback did not complete within timeout");
        }
      }
    });
  }

  void LogPlaceholderMappings() {
    Log::Write(Log::Level::Info,
               Fmt("ModelInspect: submitting with %zu placeholder mappings", placeholderMap_.size()));
    for (const auto& entry : placeholderMap_) {
      const auto placeholderAttr = entry.first->getAttribute();
      const auto globalAttr = entry.second->getAttribute();
      std::string placeholderDesc = std::holds_alternative<TensorAttribute>(placeholderAttr)
                                        ? DataTypeName(std::get<TensorAttribute>(placeholderAttr).dataType)
                                        : "GLTF";
      std::string globalDesc = std::holds_alternative<TensorAttribute>(globalAttr)
                                   ? DataTypeName(std::get<TensorAttribute>(globalAttr).dataType)
                                   : "GLTF";
      Log::Write(Log::Level::Info,
                 Fmt("  placeholder=%p (%s) -> global=%p (%s)", static_cast<void*>(entry.first.get()),
                     placeholderDesc.c_str(), static_cast<void*>(entry.second.get()), globalDesc.c_str()));
    }
  }

  void Tick() override {}

  [[nodiscard]] bool LoadingFinished() const override { return initFinished_; }

 private:
  void Initialize() {
    const std::string modelDirProp = ReadProp(kModelDirProp);
    const std::string inputProp = ReadProp(kInputFileProp);
    Log::Write(Log::Level::Info, Fmt("ModelInspect: %s=%s", kModelDirProp, modelDirProp.c_str()));
    Log::Write(Log::Level::Info, Fmt("ModelInspect: %s=%s", kInputFileProp, inputProp.c_str()));

    if (modelDirProp.empty()) {
      Log::Write(Log::Level::Error, "ModelInspect: model_dir property is empty; aborting initialization");
      FinishInit(false);
      return;
    }

    const std::filesystem::path modelDir(modelDirProp);
    const std::filesystem::path binPath = modelDir / kModelBinName;
    std::filesystem::path jsonPath = modelDir / kModelJsonName;

    if (!std::filesystem::exists(binPath)) {
      Log::Write(Log::Level::Error, Fmt("ModelInspect: missing %s", binPath.string().c_str()));
      FinishInit(false);
      return;
    }
    if (!std::filesystem::exists(jsonPath)) {
      Log::Write(Log::Level::Error, Fmt("ModelInspect: missing %s", jsonPath.string().c_str()));
      FinishInit(false);
      return;
    }

    auto jsonSpec = SecureMrUtils::LoadModelJson(jsonPath);
    if (!jsonSpec.has_value()) {
      FinishInit(false);
      return;
    }

    if (!SecureMrUtils::PrepareBindings(*jsonSpec, inputBindings_, outputBindings_, modelName_)) {
      FinishInit(false);
      return;
    }

    if (!LoadModel(binPath)) {
      FinishInit(false);
      return;
    }

    if (!SetupGlobals(inputProp)) {
      FinishInit(false);
      return;
    }

    CreatePipeline();
    SetupReadback();
    pipelineReady_ = true;
    FinishInit(true);
  }

  bool LoadModel(const std::filesystem::path& binPath) {
    std::ifstream ifs(binPath, std::ios::binary | std::ios::ate);
    if (!ifs) {
      Log::Write(Log::Level::Error, Fmt("ModelInspect: failed to open %s", binPath.string().c_str()));
      return false;
    }
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    modelBuffer_.resize(static_cast<size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(modelBuffer_.data()), size)) {
      Log::Write(Log::Level::Error, Fmt("ModelInspect: failed to read %s", binPath.string().c_str()));
      modelBuffer_.clear();
      return false;
    }
    Log::Write(Log::Level::Info, Fmt("ModelInspect: loaded model %s (%zu bytes)", binPath.string().c_str(),
                                     modelBuffer_.size()));
    return true;
  }

  bool SetupGlobals(const std::string& inputPath) {
    if (!frameworkSession_) {
      Log::Write(Log::Level::Error, "ModelInspect: framework session not ready");
      return false;
    }

    for (auto& binding : inputBindings_) {
      binding.global = std::make_shared<GlobalTensor>(frameworkSession_, binding.attr);
      LogBindingInfo("input", binding);
      if (!LoadInputData(binding, inputPath)) {
        return false;
      }
    }
    for (auto& binding : outputBindings_) {
      binding.global = std::make_shared<GlobalTensor>(frameworkSession_, binding.attr);
      LogBindingInfo("output", binding);
      const size_t bytes =
          SecureMrUtils::ElementCount(binding.attr) * SecureMrUtils::BytesPerElement(binding.attr.dataType);
      if (bytes > 0) {
        std::vector<uint8_t> zero(bytes, 0);
        binding.global->setData(reinterpret_cast<int8_t*>(zero.data()), zero.size());
      }
    }
    return true;
  }

  void LogBindingInfo(const std::string& type, const TensorBinding& binding) {
    std::string modelShape = JoinDims(binding.qnnDims);
    Log::Write(Log::Level::Info,
               Fmt("ModelInspect: %s %s (Model) shape=%s dtype=%s", type.c_str(), binding.name.c_str(),
                   modelShape.c_str(), binding.qnnType.c_str()));

    const auto elements = SecureMrUtils::ElementCount(binding.attr);
    Log::Write(Log::Level::Info,
               Fmt("ModelInspect: %s %s (SpatialML) shape=%s c=%d dtype=%s elements=%zu", type.c_str(),
                   binding.name.c_str(), JoinDims(binding.attr.dimensions).c_str(), binding.attr.channels,
                   DataTypeName(binding.attr.dataType).c_str(), elements));
  }

  static std::string JoinDims(const std::vector<int>& dims) {
    std::string out;
    for (size_t i = 0; i < dims.size(); ++i) {
      out.append(std::to_string(dims[i]));
      if (i + 1 < dims.size()) {
        out.push_back('x');
      }
    }
    if (out.empty()) {
      out = "1";
    }
    return out;
  }

  static std::string DataTypeName(XrSecureMrTensorDataTypePICO type) {
    switch (type) {
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
        return "UINT8";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO:
        return "INT8";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO:
        return "UINT16";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO:
        return "INT16";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO:
        return "INT32";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
        return "FLOAT32";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO:
        return "FLOAT64";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO:
        return "DYNAMIC_TEXTURE_UINT8";
      case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO:
        return "DYNAMIC_TEXTURE_FLOAT32";
      default:
        return "UNKNOWN";
    }
  }

  template <typename T>
  struct TensorStats {
    T min{};
    T max{};
    double mean = 0.0;
    double std = 0.0;
    size_t count = 0;
  };

  template <typename T>
  TensorStats<T> ComputeStats(const T* data, size_t count) {
    TensorStats<T> stats;
    if (count == 0) {
      return stats;
    }
    stats.count = count;
    stats.min = data[0];
    stats.max = data[0];
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
      stats.min = std::min(stats.min, data[i]);
      stats.max = std::max(stats.max, data[i]);
      sum += static_cast<double>(data[i]);
    }
    stats.mean = sum / static_cast<double>(count);

    double varSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
      const double diff = static_cast<double>(data[i]) - stats.mean;
      varSum += diff * diff;
    }
    stats.std = std::sqrt(varSum / static_cast<double>(count));
    return stats;
  }

  void LogInputPreview(const TensorBinding& binding,
                       const std::vector<uint8_t>& buffer,
                       size_t bytesUsed,
                       size_t elementBytes) {
    const size_t elements = bytesUsed / elementBytes;
    if (elements == 0) {
      Log::Write(Log::Level::Warning,
                 Fmt("ModelInspect: input %s has no elements to preview", binding.name.c_str()));
      return;
    }

    const size_t previewCount = std::min<size_t>(elements, 8);
    auto logPreview = [&](auto ptr) {
      std::string msg;
      for (size_t i = 0; i < previewCount; ++i) {
        msg.append(std::to_string(ptr[i]));
        if (i + 1 < previewCount) {
          msg.append(", ");
        }
      }
      Log::Write(Log::Level::Info,
                 Fmt("ModelInspect: input %s preview [%s]", binding.name.c_str(), msg.c_str()));
    };

    switch (binding.attr.dataType) {
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
      case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO: {
        auto* ptr = reinterpret_cast<const float*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, static_cast<double>(stats.min),
                       static_cast<double>(stats.max), stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO: {
        auto* ptr = reinterpret_cast<const double*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, stats.min, stats.max, stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO: {
        auto* ptr = reinterpret_cast<const int32_t*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, static_cast<double>(stats.min),
                       static_cast<double>(stats.max), stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO: {
        auto* ptr = reinterpret_cast<const int16_t*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, static_cast<double>(stats.min),
                       static_cast<double>(stats.max), stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO: {
        auto* ptr = reinterpret_cast<const uint16_t*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, static_cast<double>(stats.min),
                       static_cast<double>(stats.max), stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO: {
        auto* ptr = reinterpret_cast<const int8_t*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, static_cast<double>(stats.min),
                       static_cast<double>(stats.max), stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
      case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO: {
        auto* ptr = reinterpret_cast<const uint8_t*>(buffer.data());
        const auto stats = ComputeStats(ptr, elements);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s stats count=%zu min=%.6g max=%.6g mean=%.6g std=%.6g",
                       binding.name.c_str(), stats.count, static_cast<double>(stats.min),
                       static_cast<double>(stats.max), stats.mean, stats.std));
        logPreview(ptr);
        break;
      }
      default: {
        const size_t byteCount = std::min<size_t>(bytesUsed, 8);
        std::string msg;
        for (size_t i = 0; i < byteCount; ++i) {
          msg.append(std::to_string(buffer[i]));
          if (i + 1 < byteCount) {
            msg.append(", ");
          }
        }
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: input %s preview bytes [%s]", binding.name.c_str(), msg.c_str()));
        break;
      }
    }
  }

  bool LoadInputData(const TensorBinding& binding, const std::string& inputPath) {
    const size_t elementBytes = SecureMrUtils::BytesPerElement(binding.attr.dataType);
    if (elementBytes == 0) {
      Log::Write(Log::Level::Error,
                 Fmt("ModelInspect: unsupported data type %d for input %s", binding.attr.dataType, binding.name.c_str()));
      return false;
    }

    const size_t expectedBytes = SecureMrUtils::ElementCount(binding.attr) * elementBytes;
    std::vector<uint8_t> buffer(expectedBytes, 0);
    size_t bytesUsed = expectedBytes;
    const bool hasInputFile = !inputPath.empty();

    if (hasInputFile) {
      std::ifstream ifs(inputPath, std::ios::binary | std::ios::ate);
      if (!ifs) {
        Log::Write(Log::Level::Error,
                   Fmt("ModelInspect: failed to open input file %s for %s", inputPath.c_str(), binding.name.c_str()));
        return false;
      }
      std::streamsize fileSize = ifs.tellg();
      ifs.seekg(0, std::ios::beg);

      bytesUsed = std::min(static_cast<size_t>(fileSize), expectedBytes);
      if (bytesUsed > 0) {
        ifs.read(reinterpret_cast<char*>(buffer.data()), bytesUsed);
      }

      if (static_cast<size_t>(fileSize) < expectedBytes) {
        Log::Write(Log::Level::Warning,
                   Fmt("ModelInspect: input file shorter than expected for %s (%zu < %zu), padding with zeros",
                       binding.name.c_str(), static_cast<size_t>(fileSize), expectedBytes));
      }
      Log::Write(Log::Level::Info,
                 Fmt("ModelInspect: loaded input file %s for %s (%zu/%zu bytes)", inputPath.c_str(),
                     binding.name.c_str(), bytesUsed, expectedBytes));
    } else {
      FillRandom(buffer, binding.attr.dataType);
      Log::Write(Log::Level::Info,
                 Fmt("ModelInspect: generated random input for %s (%zu bytes)", binding.name.c_str(), expectedBytes));
    }

    binding.global->setData(reinterpret_cast<int8_t*>(buffer.data()), buffer.size());
    LogInputPreview(binding, buffer, bytesUsed, elementBytes);
    return true;
  }

  void FillRandom(std::vector<uint8_t>& buffer, XrSecureMrTensorDataTypePICO type) {
    std::mt19937 rng(std::random_device{}());
    if (buffer.empty()) {
      return;
    }
    Log::Write(Log::Level::Info,
               Fmt("ModelInspect: FillRandom for dtype=%s (%d), bytes=%zu",
                   DataTypeName(type).c_str(), static_cast<int>(type), buffer.size()));
    switch (type) {
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO: {
        auto* ptr = reinterpret_cast<float*>(buffer.data());
        const size_t count = buffer.size() / sizeof(float);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < count; ++i) {
          ptr[i] = dist(rng);
        }
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO: {
        auto* ptr = reinterpret_cast<double*>(buffer.data());
        const size_t count = buffer.size() / sizeof(double);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < count; ++i) {
          ptr[i] = dist(rng);
        }
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO: {
        auto* ptr = reinterpret_cast<int32_t*>(buffer.data());
        const size_t count = buffer.size() / sizeof(int32_t);
        std::uniform_int_distribution<int32_t> dist(-128, 128);
        for (size_t i = 0; i < count; ++i) {
          ptr[i] = dist(rng);
        }
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO: {
        auto* ptr = reinterpret_cast<int16_t*>(buffer.data());
        const size_t count = buffer.size() / sizeof(int16_t);
        std::uniform_int_distribution<int16_t> dist(-128, 128);
        for (size_t i = 0; i < count; ++i) {
          ptr[i] = dist(rng);
        }
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO: {
        std::uniform_int_distribution<int16_t> dist(-16, 16);
        for (auto& b : buffer) {
          b = static_cast<uint8_t>(dist(rng));
        }
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO: {
        auto* ptr = reinterpret_cast<uint16_t*>(buffer.data());
        const size_t count = buffer.size() / sizeof(uint16_t);
        std::uniform_int_distribution<uint16_t> dist(0, 255);
        for (size_t i = 0; i < count; ++i) {
          ptr[i] = dist(rng);
        }
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
      case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO: {
        std::uniform_int_distribution<uint16_t> dist(0, 255);
        for (auto& b : buffer) {
          b = static_cast<uint8_t>(dist(rng));
        }
        break;
      }
      default:
        break;
    }
  }

  void CreatePipeline() {
    pipeline_ = std::make_shared<Pipeline>(frameworkSession_);
    std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> inputPlaceholders;
    std::unordered_map<std::string, std::shared_ptr<PipelineTensor>> outputPlaceholders;

    for (auto& binding : inputBindings_) {
      auto placeholder = PipelineTensor::PipelinePlaceholderLike(pipeline_, binding.global);
      inputPlaceholders.emplace(binding.name, placeholder);
      placeholderMap_.emplace(placeholder, binding.global);
      Log::Write(Log::Level::Info,
                 Fmt("ModelInspect: placeholder created for input %s (placeholder=%p, global=%p)",
                     binding.name.c_str(), static_cast<void*>(placeholder.get()),
                     static_cast<void*>(binding.global.get())));
    }
    for (auto& binding : outputBindings_) {
      auto placeholder = PipelineTensor::PipelinePlaceholderLike(pipeline_, binding.global);
      outputPlaceholders.emplace(binding.name, placeholder);
      placeholderMap_.emplace(placeholder, binding.global);
      Log::Write(Log::Level::Info,
                 Fmt("ModelInspect: placeholder created for output %s (placeholder=%p, global=%p)",
                     binding.name.c_str(), static_cast<void*>(placeholder.get()),
                     static_cast<void*>(binding.global.get())));
    }

    pipeline_->runAlgorithm(reinterpret_cast<char*>(modelBuffer_.data()), modelBuffer_.size(), inputPlaceholders, {},
                            outputPlaceholders, {}, modelName_);

    Log::Write(Log::Level::Info, Fmt("ModelInspect: pipeline created for model %s", modelName_.c_str()));
  }

  void SetupReadback() {
    if (outputBindings_.empty()) {
      return;
    }
    std::vector<TensorReadback::Target> targets;
    targets.reserve(outputBindings_.size());
    for (auto& binding : outputBindings_) {
      targets.push_back(TensorReadback::Target{
          .tensor = binding.global,
          .callback = [this, name = binding.name](TensorReadbackResult&& result) { HandleReadback(name, std::move(result)); },
          .name = binding.name});
    }
    TensorReadback::Config cfg{.pollingInterval = std::chrono::milliseconds(50)};
    readback_ = std::make_unique<TensorReadback>(frameworkSession_, std::move(targets), cfg);
    remainingOutputs_.store(static_cast<int>(outputBindings_.size()));
  }

  void HandleReadback(const std::string& name, TensorReadbackResult&& result) {

    const auto outPath = outputDir_ / Fmt("model_inspect_output_%s.bin", name.c_str());
    std::ofstream ofs(outPath, std::ios::binary);
    if (!ofs) {
      Log::Write(Log::Level::Error, Fmt("ModelInspect: failed to open %s for write", outPath.string().c_str()));
    } else {
      ofs.write(reinterpret_cast<const char*>(result.data.data()),
                static_cast<std::streamsize>(result.data.size()));
      ofs.flush();
      const auto wroteBytes = static_cast<size_t>(ofs.tellp());
      if (!ofs.good()) {
        Log::Write(Log::Level::Error, Fmt("ModelInspect: write error for %s", outPath.string().c_str()));
      }
      Log::Write(Log::Level::Info,
                 Fmt("ModelInspect: dumped %s (%zu bytes) -> %s", name.c_str(), result.data.size(),
                     outPath.string().c_str()));
      std::error_code ec;
      const auto onDiskBytes = std::filesystem::file_size(outPath, ec);
      if (!ec && static_cast<size_t>(onDiskBytes) != result.data.size()) {
        Log::Write(Log::Level::Warning,
                   Fmt("ModelInspect: file_size mismatch for %s (expected %zu, tellp %zu, on-disk %lld)",
                       outPath.string().c_str(), result.data.size(), wroteBytes,
                       static_cast<long long>(onDiskBytes)));
      }
    }

    auto logPreview = [&](auto ptr, size_t count) {
      std::string msg;
      for (size_t i = 0; i < count; ++i) {
        msg.append(std::to_string(ptr[i]));
        if (i + 1 < count) {
          msg.append(", ");
        }
      }
      Log::Write(Log::Level::Info, Fmt("ModelInspect: %s preview [%s]", name.c_str(), msg.c_str()));
    };

    switch (result.dataType) {
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO: {
        auto* ptr = reinterpret_cast<const float*>(result.data.data());
        const size_t floatCount = result.data.size() / sizeof(float);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: %s has %zu float32 values", name.c_str(), floatCount));
        const size_t count = std::min<size_t>(floatCount, 8);
        logPreview(ptr, count);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO: {
        auto* ptr = reinterpret_cast<const double*>(result.data.data());
        const size_t floatCount = result.data.size() / sizeof(double);
        Log::Write(Log::Level::Info,
                   Fmt("ModelInspect: %s has %zu float64 values", name.c_str(), floatCount));
        const size_t count = std::min<size_t>(floatCount, 8);
        logPreview(ptr, count);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO: {
        auto* ptr = reinterpret_cast<const int32_t*>(result.data.data());
        const size_t count = std::min<size_t>(result.data.size() / sizeof(int32_t), 8);
        logPreview(ptr, count);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO: {
        auto* ptr = reinterpret_cast<const int16_t*>(result.data.data());
        const size_t count = std::min<size_t>(result.data.size() / sizeof(int16_t), 8);
        logPreview(ptr, count);
        break;
      }
      case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO: {
        auto* ptr = reinterpret_cast<const uint16_t*>(result.data.data());
        const size_t count = std::min<size_t>(result.data.size() / sizeof(uint16_t), 8);
        logPreview(ptr, count);
        break;
      }
      default: {
        const size_t count = std::min<size_t>(result.data.size(), 8);
        logPreview(result.data.data(), count);
        break;
      }
    }

    if (remainingOutputs_.fetch_sub(1) == 1) {
      readbackDone_ = true;
      if (readback_) {
        readback_->Stop();
      }
      readbackCv_.notify_all();
    }
  }

  void FinishInit(bool ok) {
    {
      std::lock_guard<std::mutex> lock(initMutex_);
      initFinished_ = true;
      if (!ok) {
        Log::Write(Log::Level::Error, "ModelInspect: initialization failed");
      } else {
        Log::Write(Log::Level::Info, "ModelInspect: initialization finished");
      }
    }
    initCv_.notify_all();
  }

  XrInstance xrInstance_;
  XrSession xrSession_;

  std::shared_ptr<FrameworkSession> frameworkSession_;
  std::shared_ptr<Pipeline> pipeline_;
  std::vector<uint8_t> modelBuffer_;
  std::vector<TensorBinding> inputBindings_;
  std::vector<TensorBinding> outputBindings_;
  std::map<std::shared_ptr<PipelineTensor>, std::shared_ptr<GlobalTensor>> placeholderMap_;

  std::unique_ptr<TensorReadback> readback_;
  std::filesystem::path outputDir_;

  std::unique_ptr<std::thread> initThread_;
  std::unique_ptr<std::thread> runThread_;
  std::mutex initMutex_;
  std::condition_variable initCv_;
  bool initFinished_ = false;
  bool pipelineReady_ = false;

  std::mutex readbackMutex_;
  std::condition_variable readbackCv_;
  std::atomic<bool> readbackDone_{false};
  std::atomic<int> remainingOutputs_{0};

  std::string modelName_ = "model_inspect";
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<ModelInspectApp>(instance, session);
}

}  // namespace SecureMR
