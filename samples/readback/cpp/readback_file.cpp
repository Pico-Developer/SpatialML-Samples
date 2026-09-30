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

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "readback_file.h"
#include <string>
#include <sstream>
#include <android/log.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_com_bytedance_pico_spatial_1ml_1demo_readback_ReadbackActivity_nativeSetPermission(
    JNIEnv* env,
    jclass,
    jstring permission,
    jboolean granted)
{
  const char* permUtf = env->GetStringUTFChars(permission, nullptr);
  std::string perm(permUtf);
  env->ReleaseStringUTFChars(permission, permUtf);

  bool isGranted = (granted == JNI_TRUE);

  {
    std::lock_guard<std::mutex> lock(g_permMutex);
    if (perm == "android.permission.CAMERA")
      gPermissionCamera = isGranted;
  }
}

#ifdef __cplusplus
}
#endif

namespace SecureMR {

#ifdef XR_READBACK_USE_CPU
bool ReadbackCheck::isCpuBuffer = true;
#else
bool ReadbackCheck::isCpuBuffer = false;
#endif

struct android_app* ReadbackCheck::gapp = nullptr;

ReadbackCheck::ReadbackCheck(const XrInstance& instance, const XrSession& session)
    : xr_instance(instance), xr_session(session) {}

ReadbackCheck::~ReadbackCheck() {
  keepRunning = false;
  if (readbackTexture != XR_NULL_HANDLE)
  {
    mReadbackController->ReleaseReadbackTexture(readbackTexture);
  }

  delete mReadbackController;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto& runner : pipelineRunners) {
    if (runner.joinable()) runner.join();
  }
}

void ReadbackCheck::CreateFramework() {
  LOGI("Request Permission");
  Log::Write(Log::Level::Info, "CreateFramework ...");
  frameworkSession = std::make_shared<FrameworkSession>(xr_instance, xr_session, 512, 512);
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void ReadbackCheck::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    // Note: global tensors must be created before they are referred
    //       in each individual pipeline
    CreateGlobalTensor();
    CreateRelaxMrReadBackPipeline();
    initialized.notify_all();
    pipelineAllInitialized = true;
  });

}

void ReadbackCheck::CreateGlobalTensor() {
  Log::Write(Log::Level::Info, "CreateGlobalTensor ...");
  if (isCpuBuffer)
  {
    vstOutputLeftUint8Global = std::make_shared<GlobalTensor>(
       frameworkSession,
       TensorAttribute{.dimensions = {512, 512}, .channels = 3, .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
    assert(vstOutputLeftUint8Global != nullptr);
  }
  else {
    vstOutputLeftUint8Global = std::make_shared<GlobalTensor>(
        frameworkSession, TensorAttribute{.dimensions = {512, 512},
                                          .channels = 3,
                                          .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_DYNAMIC_TEXTURE_PICO,
                                          .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO});

    assert(vstOutputLeftUint8Global != nullptr);
  }
  initializeGraphicsContext();
  mConfig.w = 512; mConfig.h = 512;
  mReadbackController = new ReadbackController(frameworkSession, vstOutputLeftUint8Global);
  mCurrentReadbackRequest = nullptr;
  Log::Write(Log::Level::Info, "CreateGlobalTensor Done...");
}


void ReadbackCheck::Tick()
{

  if (!pipelineAllInitialized || !gPermissionCamera) return;

  if (isCpuBuffer)
  {
    // cpu buffer
    if (!mCurrentReadbackRequest) {
      mReadbackController->RequestReadbackBuffer(mCurrentReadbackRequest);
    }
    else {
      auto result = new XrReadbackTensorBufferPICO();
      if (mReadbackController->TryAcquireReadbackBuffer(*mCurrentReadbackRequest, result)) {
        mCurrentReadbackRequest = nullptr;
        std::string path = gapp->activity->externalDataPath;
        path = path + "/output.png";
        OutputReadbackBufferToFile(result, path);
        delete((char*)result->buffer);
      }
      delete result;
    }
    
  }
  else
  {
    // hardware texture
    if (!mCurrentReadbackRequest) {
       mReadbackController->RequestReadbackTexture(mCurrentReadbackRequest);
    }
    else
    {
      if (readbackTexture == XR_NULL_HANDLE) {
        if (mReadbackController->TryAcquireReadbackTexture(*mCurrentReadbackRequest, readbackTexture)) {
          mCurrentReadbackRequest = nullptr;
        }
      }
      else
      {
        std::string path = gapp->activity->externalDataPath;
        path = path + "/output.png";
        OutputReadbackTextureToPath(readbackTexture, path);
      }
    }
  }
}

void ReadbackCheck::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunRelaxMrReadBackPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });
}

void ReadbackCheck::RequestPermission(struct android_app* app) {
  LOGI("RelaxMR Request Permission");
  ReadbackCheck::gapp = app;
  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);

  jobject activity = app->activity->clazz;        
  jclass  cls      = env->GetObjectClass(activity);
  jmethodID mid    = env->GetMethodID(cls, "requestCameraFromNative", "()V");
  env->CallVoidMethod(activity, mid);
}
void ReadbackCheck::OutputReadbackBufferToFile(const XrReadbackTensorBufferPICO* tensorBuffer, const std::string &path) {
  ImageRGB img;
  img.rgb.resize(tensorBuffer->bufferCapacityInput);
  std::memcpy(img.rgb.data(), tensorBuffer->buffer, tensorBuffer->bufferCapacityInput);
  img.w = mConfig.w;
  img.h = mConfig.h;
  LOGI("output path %s", path.c_str());
  stbi_write_png(path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3);
}

void ReadbackCheck::CreateRelaxMrReadBackPipeline() {
  LOGI("RelaxMR CreateRelaxMrReadBackPipeline");
  m_RelaxMrReadBackPipeline = std::make_shared<Pipeline>(frameworkSession);
  vstOutputLeftUint8Placeholder =
      PipelineTensor::PipelinePlaceholderLike(m_RelaxMrReadBackPipeline, vstOutputLeftUint8Global);
  m_RelaxMrReadBackPipeline->cameraAccess(nullptr, vstOutputLeftUint8Placeholder, nullptr, nullptr);
}

XrSecureMrPipelineRunPICO ReadbackCheck::RunRelaxMrReadBackPipeline(const XrSecureMrPipelineRunPICO pre) {
  return  m_RelaxMrReadBackPipeline->submit({{vstOutputLeftUint8Placeholder, vstOutputLeftUint8Global}},
                                            XR_NULL_HANDLE, nullptr);
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<ReadbackCheck>(instance, session);
}
}  // namespace SecureMR
