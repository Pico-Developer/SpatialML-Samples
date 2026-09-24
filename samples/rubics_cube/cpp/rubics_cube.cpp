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

#include "rubics_cube.h"

#include <android_native_app_glue.h>
#include <jni.h>
#include <utility>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <random>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"
#include "camera_manager.h"
#include "color_classifier.h"
#include "cube_model.h"
#include "logger.h"
#include "scan_manager.h"
#include "solver.h"
#include "scan_ui.h"
#include "virtual_cube.h"
#include "xr_linear.h"

namespace SecureMR {

extern AAssetManager* g_assetManager;

struct android_app* RubicsCube::gapp = nullptr;

namespace {
constexpr auto kReadbackInterval = std::chrono::milliseconds(50);
constexpr float kCubeForwardOffsetM = 0.85f;
constexpr float kCubeLeftOffsetM = 0.25f;

ScanUiState::Arrow ArrowForStep(int step) {
  static const ScanUiState::Arrow seq[6] = {ScanUiState::Arrow::Down, ScanUiState::Arrow::Right,
                                            ScanUiState::Arrow::Down, ScanUiState::Arrow::Right,
                                            ScanUiState::Arrow::Down, ScanUiState::Arrow::Right};
  if (step <= 0) return ScanUiState::Arrow::None;
  int idx = step - 1;
  if (idx >= 0 && idx < 6) return seq[idx];
  return ScanUiState::Arrow::None;
}

std::optional<VirtualCube::CubeTurnOp> TurnForArrow(ScanUiState::Arrow arrow) {
  switch (arrow) {
    case ScanUiState::Arrow::Right: return VirtualCube::CubeTurnOp::Right;
    case ScanUiState::Arrow::Down: return VirtualCube::CubeTurnOp::Down;
    default: return std::nullopt;
  }
}

template <typename T>
std::array<T, 9> RemapGrid(const std::array<T, 9>& input, const std::array<int, 9>& mapping) {
  std::array<T, 9> output{};
  for (int i = 0; i < 9; ++i) {
    int target = mapping[i];
    if (target >= 0 && target < 9) {
      output[target] = input[i];
    }
  }
  return output;
}

Color LetterToColor(char letter) {
  switch (letter) {
    case 'U': return Color::W;
    case 'R': return Color::R;
    case 'F': return Color::G;
    case 'D': return Color::Y;
    case 'L': return Color::O;
    case 'B': return Color::B;
    default: return Color::Unknown;
  }
}

PatchColor ColorToPatch(Color color) {
  switch (color) {
    case Color::W: return {1.0f, 1.0f, 1.0f};
    case Color::Y: return {1.0f, 1.0f, 0.0f};
    case Color::R: return {1.0f, 0.0f, 0.0f};
    case Color::O: return {1.0f, 0.55f, 0.0f};
    case Color::G: return {0.0f, 1.0f, 0.0f};
    case Color::B: return {0.0f, 0.0f, 1.0f};
    default: return {0.3f, 0.3f, 0.3f};
  }
}
}  // namespace

RubicsCube::RubicsCube(const XrInstance& instance, const XrSession& session)
    : instance_(instance), session_(session) {}

RubicsCube::~RubicsCube() {
  if (readback_) {
    readback_->Stop();
  }
  if (scanUi_) {
    scanUi_->StopThread();
  }
}

void RubicsCube::CreateFramework() {
  Log::SetLevel(Log::Level::Debug);
  Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: Initializing SpatialML framework session for readback sample");
  const int imgW = overlayW_ > 0 ? overlayW_ : 1024;
  const int imgH = overlayH_ > 0 ? overlayH_ : 1024;
  frameworkSession_ = std::make_shared<FrameworkSession>(instance_, session_, imgW, imgH);
  Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: Framework session created");
  
  // Initialize components
  cameraManager_ = std::make_unique<CameraManager>();
  colorClassifier_ = std::make_shared<ColorClassifier>();
  scanManager_ = std::make_unique<ScanManager>(colorClassifier_);
  cubeModel_ = std::make_unique<CubeModel>();
  solver_ = std::make_unique<Solver>();
  virtualCube_ = std::make_unique<VirtualCube>();
  scanUi_ = std::make_unique<ScanUi>();

  if (scanUi_ && scanManager_) {
    scanUi_->SetScanManager(scanManager_.get());
    // Start UI composition thread with initial overlay size hint
    const int imgW = overlayW_ > 0 ? overlayW_ : 1024;
    const int imgH = overlayH_ > 0 ? overlayH_ : 1024;
    scanUi_->SetOverlaySize(imgW, imgH);
    scanUi_->StartThread();
  }
  stage_ = AppStage::Scan;
  hasLastAcceptedCenter_ = false;
  solutionText_.clear();
  solutionSteps_.clear();
  cachedFacelets_.reset();
  scanIndex_ = 0;
  Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: Managers created; entering Scan stage");
  if (virtualCube_ && cubeModel_) virtualCube_->SetCubeModel(cubeModel_.get());
  if (virtualCube_) {
    virtualCube_->ResetOrientation();
  }
  ClearArrowHold();
}

void RubicsCube::CreateGlobalTensors() {
  const int imgW = overlayW_ > 0 ? overlayW_ : 1024;
  const int imgH = overlayH_ > 0 ? overlayH_ : 1024;
  vstOutputLeftUint8Global_ = std::make_shared<GlobalTensor>(
      frameworkSession_,
      TensorAttribute{.dimensions = {imgW, imgH},
                      .channels = 3,
                      .usage = XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                      .dataType = XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO});
}

void RubicsCube::CreateRelaxMrReadbackPipeline() {
  relaxMrReadbackPipeline_ = std::make_shared<Pipeline>(frameworkSession_);
  vstOutputLeftUint8Placeholder_ =
      PipelineTensor::PipelinePlaceholderLike(relaxMrReadbackPipeline_, vstOutputLeftUint8Global_);
  relaxMrReadbackPipeline_->cameraAccess(nullptr, vstOutputLeftUint8Placeholder_, nullptr, nullptr);
}

void RubicsCube::CreatePipelines() {
  CreateGlobalTensors();
  CreateRelaxMrReadbackPipeline();

  TensorReadback::Target target{
      .tensor = vstOutputLeftUint8Global_,
      .callback = [this](TensorReadbackResult&& result) { HandleReadbackResult(std::move(result)); },
      .name = "vst_output_left_u8"};

  TensorReadback::Config config{.pollingInterval = std::chrono::milliseconds(33)};
  readback_ = std::make_unique<TensorReadback>(frameworkSession_, std::vector<TensorReadback::Target>{std::move(target)}, config);
  readback_->Start();

  pipelinesInitialized_ = true;
  Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: Pipelines and readback utilities initialized");
}

void RubicsCube::RunRelaxMrReadbackPipeline() {
  if (!relaxMrReadbackPipeline_) {
    return;
  }
  relaxMrReadbackPipeline_->submit({{vstOutputLeftUint8Placeholder_, vstOutputLeftUint8Global_}}, XR_NULL_HANDLE, nullptr);
}

void RubicsCube::Tick() {
  if (!pipelinesInitialized_) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (stage_ != AppStage::Scan && arrowHoldActive_) {
    ClearArrowHold();
  } else if (arrowHoldActive_) {
    if (now - arrowHoldStart_ >= arrowHoldDuration_) {
      ClearArrowHold();
    }
  }
  // Only submit readback during scanning to save bandwidth/CPU in later stages
  if (stage_ == AppStage::Scan && now - lastRelaxReadbackRunTime_ >= kReadbackInterval) {
    RunRelaxMrReadbackPipeline();
    lastRelaxReadbackRunTime_ = now;
  }
  if (awaitingValidate_) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - scanCompleteTime_);
    if (elapsed >= std::chrono::seconds(1)) {
      awaitingValidate_ = false;
      stage_ = AppStage::Validate;
      solutionText_.clear();
      solutionSteps_.clear();
      Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: delay elapsed; entering Validate stage");
      ClearArrowHold();
    }
  }
  if (now - lastTickLog_ >= std::chrono::seconds(1)) {
    auto stageName = [](AppStage s) -> const char* {
      switch (s) {
        case AppStage::Idle: return "Idle";
        case AppStage::Scan: return "Scan";
        case AppStage::Validate: return "Validate";
        case AppStage::Solve: return "Solve";
        case AppStage::Animate: return "Animate";
        case AppStage::Done: return "Done";
        default: return "Unknown";
      }
    };
    Log::Write(Log::Level::Info,
               Fmt("RubicsCube|TickFlow: stage=%s scanIndex=%d awaitingValidate=%d arrowHold=%d lastCenter=%d cachedFacelets=%d headPose=%d",
                   stageName(stage_),
                   scanIndex_,
                   awaitingValidate_ ? 1 : 0,
                   arrowHoldActive_ ? 1 : 0,
                   hasLastAcceptedCenter_ ? 1 : 0,
                   cachedFacelets_.has_value() ? 1 : 0,
                   hasHeadPose_ ? 1 : 0));
    lastTickLog_ = now;
  }
  // continue into scanning state machine

  // Lightweight state machine driver
  if (stage_ == AppStage::Scan && cameraManager_ && scanManager_) {
    if (scanIndex_ >= static_cast<int>(scanOrder_.size())) {
      // All faces have been scanned; wait for the delayed Validate transition without touching scanOrder_.
      return;
    }
    auto frameOpt = cameraManager_->GetLatestFrame();
    if (frameOpt) {
      Face target = scanOrder_[scanIndex_];
      Face previewFace = target;
      if (auto facing = GetFacingFaceFromHeadPose()) {
        previewFace = *facing;
      }
      const bool allowFaceOps = !arrowHoldActive_;
      if (allowFaceOps && virtualCube_) {
        auto patches = scanManager_->GetPatchColors9U8();
        std::array<Color, 9> preview{};
        std::array<XrVector3f, 9> previewDisplay{};
        auto classifyColor = [&](uint8_t r8, uint8_t g8, uint8_t b8) {
          PatchColor pc{r8 / 255.0f, g8 / 255.0f, b8 / 255.0f};
          if (colorClassifier_) {
            return colorClassifier_->Classify(pc);
          }
          if (r8 >= g8 && r8 >= b8) return Color::R;
          if (g8 >= r8 && g8 >= b8) return Color::G;
          if (b8 >= r8 && b8 >= g8) return Color::B;
          return Color::Unknown;
        };
        for (int i = 0; i < 9; ++i) {
          preview[i] = classifyColor(patches[i][0], patches[i][1], patches[i][2]);
          previewDisplay[i] = {patches[i][0] / 255.0f, patches[i][1] / 255.0f, patches[i][2] / 255.0f};
        }
        if (hasHeadPose_) {
          std::array<int, 9> mapping{};
          if (virtualCube_->MapScreenGridToFaceIndices(previewFace, lastHeadPose_.orientation, mapping)) {
            preview = RemapGrid(preview, mapping);
            previewDisplay = RemapGrid(previewDisplay, mapping);
          }
        }
        virtualCube_->PreviewFace(previewFace, preview, previewDisplay);
      }
      Log::Write(Log::Level::Debug,
                 Fmt("RubicsCube|Orchestrator: Tick scanning index=%d targetFace=%d", scanIndex_, static_cast<int>(target)));
      if (allowFaceOps && scanManager_->TryLockFace(target, *frameOpt)) {
        auto locked = scanManager_->GetLockedFace();
        if (locked && cubeModel_ && colorClassifier_) {
          Face facingFace = target;
          if (auto facing = GetFacingFaceFromHeadPose()) {
            facingFace = *facing;
          }
          locked->face = facingFace;
          if (hasHeadPose_ && virtualCube_) {
            std::array<int, 9> mapping{};
            if (virtualCube_->MapScreenGridToFaceIndices(facingFace, lastHeadPose_.orientation, mapping)) {
              auto reordered = RemapGrid(locked->patches, mapping);
              locked->patches = reordered;
            }
          }
          // Compute 8-bit center color of the newly locked face
          const auto& pc = locked->patches[4];
          auto clampU8 = [](int v) { return static_cast<uint8_t>(std::min(std::max(v, 0), 255)); };
          const uint8_t cr = clampU8(static_cast<int>(std::round(pc.r * 255.0f)));
          const uint8_t cg = clampU8(static_cast<int>(std::round(pc.g * 255.0f)));
          const uint8_t cb = clampU8(static_cast<int>(std::round(pc.b * 255.0f)));

          bool accept = true;
          if (hasLastAcceptedCenter_) {
            const int tol = 20;  // per-channel tolerance
            const int dr = std::abs((int)cr - (int)lastAcceptedCenterU8_[0]);
            const int dg = std::abs((int)cg - (int)lastAcceptedCenterU8_[1]);
            const int db = std::abs((int)cb - (int)lastAcceptedCenterU8_[2]);
            if (dr <= tol && dg <= tol && db <= tol) {
              accept = false;  // same center color as previous accepted face -> do not advance
            }
          }

          if (!accept) {
            Log::Write(Log::Level::Warning,
                       Fmt("RubicsCube|Orchestrator: repeated face center color; clearing cube model and restarting scan"));
            if (virtualCube_) {
              virtualCube_->ClearPreviewFace(facingFace);
              virtualCube_->ClearPreview();
            }
            if (cubeModel_) cubeModel_->Clear();
            scanIndex_ = 0;
            hasLastAcceptedCenter_ = false;
            solutionText_.clear();
            solutionSteps_.clear();
            cachedFacelets_.reset();
            stage_ = AppStage::Scan;
            if (virtualCube_) {
              virtualCube_->SetCubeModel(cubeModel_.get());
              virtualCube_->ClearPreviewFace(scanOrder_[0]);
              virtualCube_->ResetOrientation();
            }
            awaitingValidate_ = false;
            ClearArrowHold();
          } else {
            cubeModel_->SetFaceFromPatches(*locked, *colorClassifier_);
            if (virtualCube_) {
              virtualCube_->ClearPreviewFace(facingFace);
              virtualCube_->SetCubeModel(cubeModel_.get());
            }
            cachedFacelets_.reset();
            lastAcceptedCenterU8_ = {cr, cg, cb};
            hasLastAcceptedCenter_ = true;
            scanIndex_++;
            Log::Write(Log::Level::Info, Fmt("RubicsCube|Orchestrator: face %d captured; progress %d/6",
                                             static_cast<int>(target), scanIndex_));
            // Pause scanning for 500ms to avoid cross-face crosstalk
            if (scanManager_) scanManager_->SetPauseForMs(500);
            if (scanIndex_ >= 6 && !awaitingValidate_) {
              awaitingValidate_ = true;
              scanCompleteTime_ = std::chrono::steady_clock::now();
              Log::Write(Log::Level::Info,
                         "RubicsCube|Orchestrator: all faces scanned; holding for 1s before Validate");
              if (virtualCube_) virtualCube_->ClearPreview();
            }
            if (virtualCube_) {
              auto nextArrow = ArrowForStep(scanIndex_);
              if (auto turn = TurnForArrow(nextArrow)) {
                StartArrowHold(nextArrow, now);
                if (hasHeadPose_) {
                  virtualCube_->QueueOrientationOpRelative(*turn, lastHeadPose_.orientation);
                } else {
                  virtualCube_->QueueOrientationOp(*turn);
                }
              } else {
                ClearArrowHold();
              }
            }
          }
        }
      }
    }
  }

  // Update overlay Scan UI each tick (static dashed box + arrow + center color)
  if ((stage_ == AppStage::Scan || stage_ == AppStage::Validate || stage_ == AppStage::Solve) && scanUi_) {
    const auto now2 = std::chrono::steady_clock::now();
    float dt = 0.016f;
    if (lastUiTick_.time_since_epoch().count() != 0) {
      dt = std::chrono::duration_cast<std::chrono::milliseconds>(now2 - lastUiTick_).count() / 1000.0f;
    }
    lastUiTick_ = now2;
    if (virtualCube_) {
      if (stage_ == AppStage::Solve) {
        virtualCube_->Solving();
      } else {
        virtualCube_->Update(dt);
      }
    }

    ScanUiState ui{};
    ui.scannedCount = cubeModel_ ? cubeModel_->GetScannedCount() : 0;
    ui.activeSolveIndex = -1;
    ui.solveSteps.clear();
    // Arrow guidance only during Scan
    if (stage_ == AppStage::Scan && arrowHoldActive_) {
      ui.nextArrow = arrowHoldArrow_;
    } else {
      ui.nextArrow = ScanUiState::Arrow::None;
    }

    // Phase and bottom text
    if (stage_ == AppStage::Scan) {
      ui.phase = ScanUiState::Phase::Stabilizing;
      bool showMotion = false;
      if (lastMotionReset_.time_since_epoch().count() != 0) {
        showMotion = (now2 - lastMotionReset_) < std::chrono::milliseconds(1500);
      }
      if (showMotion) {
        ui.bottomText = std::string("Motion detected, restarting scan ...");
      } else if (ui.scannedCount <= 0) {
        ui.bottomText = std::string("Please place cube inside box.");
      } else {
        ui.bottomText = Fmt("Scanned %d/6 ...", ui.scannedCount);
      }
    } else if (stage_ == AppStage::Validate) {
      ui.phase = ScanUiState::Phase::Validate;
      ui.bottomText = std::string("Hold steady for solution...");
    } else if (stage_ == AppStage::Solve) {
      ui.phase = ScanUiState::Phase::Solve;
      ui.bottomText = solutionText_.empty() ? std::string("Solving ...") : solutionText_;
      ui.solveSteps = solutionSteps_;
      if (virtualCube_) {
        if (auto idx = virtualCube_->ActiveMoveIndex()) {
          if (*idx >= 0 && *idx < static_cast<int>(solutionSteps_.size())) {
            ui.activeSolveIndex = *idx;
          }
        }
      }
    }

    // Provide hint colors (first face) for Validate/Solve
    if (stage_ == AppStage::Validate || stage_ == AppStage::Solve) {
      auto toRgb = [](Color c) -> std::array<uint8_t, 3> {
        switch (c) {
          case Color::W: return {255, 255, 255};
          case Color::Y: return {255, 255, 0};
          case Color::R: return {255, 0, 0};
          case Color::O: return {255, 140, 0};
          case Color::G: return {0, 200, 0};
          case Color::B: return {0, 100, 255};
          default: return {80, 80, 80};
        }
      };
      if (cubeModel_) {
        auto faceColorsOpt = cubeModel_->GetFaceColors(Face::U);
        if (faceColorsOpt) {
          for (int i = 0; i < 9; ++i) ui.hintColors9[i] = toRgb((*faceColorsOpt)[i]);
        }
      }
    }
    auto frameOpt2 = cameraManager_ ? cameraManager_->GetLatestFrame() : std::optional<Frame>{};
    scanUi_->Update(frameOpt2 ? &*frameOpt2 : nullptr, ui, dt);
  }

  if (stage_ == AppStage::Validate) {
    if (!cachedFacelets_) {
      cachedFacelets_ = BuildFaceletsFromVirtualCube();
      if (cachedFacelets_) {
        if (!UpdateCubeModelFromFacelets(*cachedFacelets_)) {
          Log::Write(Log::Level::Warning, "VirtualCube: failed to apply cube model from rebuilt facelets during Validate");
        }
      } else {
        Log::Write(Log::Level::Warning, "VirtualCube: BuildFaceletsFromVirtualCube() returned empty result during Validate");
      }
    }
    if (cachedFacelets_ && ValidateFacelets(*cachedFacelets_)) {
      stage_ = AppStage::Solve;
      solutionText_.clear();
      solutionSteps_.clear();
      Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: validation passed; entering Solve stage");
      ClearArrowHold();
      if (virtualCube_) virtualCube_->ClearPreview();
    } else {
      // If invalid, reset to rescan
      if (virtualCube_) {
        if (scanIndex_ >= 0 && scanIndex_ < static_cast<int>(scanOrder_.size())) {
          virtualCube_->ClearPreviewFace(scanOrder_[scanIndex_]);
        }
        virtualCube_->ClearPreview();
      }
      if (cubeModel_) cubeModel_->Clear();
      scanIndex_ = 0;
      hasLastAcceptedCenter_ = false;
      solutionText_.clear();
      solutionSteps_.clear();
      cachedFacelets_.reset();
      stage_ = AppStage::Scan;
      Log::Write(Log::Level::Warning, "RubicsCube|Orchestrator: validation failed; restarting scan");
      if (virtualCube_) {
        virtualCube_->SetCubeModel(cubeModel_.get());
        virtualCube_->ClearPreviewFace(scanOrder_[0]);
        virtualCube_->ClearPreview();
        virtualCube_->ResetOrientation();
      }
      awaitingValidate_ = false;
      ClearArrowHold();
    }
  }

  if (stage_ == AppStage::Solve && cubeModel_ && solver_) {
    if (solutionText_.empty()) {
      if (cachedFacelets_) {
        Log::Write(Log::Level::Info,
                   Fmt("VirtualCube: cached facelets ready for Solve (len=%zu)", cachedFacelets_->size()));
        if (!UpdateCubeModelFromFacelets(*cachedFacelets_)) {
          Log::Write(Log::Level::Warning, "VirtualCube: failed to refresh cube model from cached facelets prior to Solve");
        }
      } else {
        Log::Write(Log::Level::Warning, "VirtualCube: Solve entered without cached facelets; using CubeModel snapshot");
      }
      const std::string code = cachedFacelets_.value_or(cubeModel_->ToKociembaFacelets());
      Log::Write(Log::Level::Info, Fmt("RubicsCube|Orchestrator: invoking solver with facelets=%s", code.c_str()));
      solutionText_ = solver_->Solve(code);
      solutionSteps_.clear();
      if (solutionText_.empty()) {
        solutionText_ = std::string("Solve failed (invalid cube or timeout)");
        Log::Write(Log::Level::Warning, "RubicsCube|Orchestrator: solver returned empty result; staying in Solve stage");
      } else {
        Log::Write(Log::Level::Info, Fmt("RubicsCube|Orchestrator: solution steps: %s", solutionText_.c_str()));
        // Parse and queue moves to the renderer
        if (virtualCube_) {
          std::vector<Move> moves;
          moves.reserve(64);
          auto tokenize = [](const std::string& s) {
            std::vector<std::string> toks; std::string cur;
            for (char ch : s) {
              if (ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r') { if (!cur.empty()) { toks.push_back(cur); cur.clear(); } }
              else { cur.push_back(ch); }
            }
            if (!cur.empty()) toks.push_back(cur);
            return toks;
          };
          auto toks = tokenize(solutionText_);
          solutionSteps_ = toks;
          for (auto& t : toks) {
            if (t.empty()) continue;
            Move m{};
            switch (t[0]) {
              case 'U': m.type = MoveType::U; break;
              case 'D': m.type = MoveType::D; break;
              case 'F': m.type = MoveType::F; break;
              case 'B': m.type = MoveType::B; break;
              case 'R': m.type = MoveType::L; break;  // swap L and R for virtual cube
              case 'L': m.type = MoveType::R; break;
              default: continue;
            }
            if (t.size() >= 2) {
              if (t[1] == '\'') { m.turn = Turn::CCW; }
              else if (t[1] == '2') { m.turn = Turn::Double; }
            }
          
            // Fix B/F
            if (m.type == MoveType::B ||
                m.type == MoveType::F
                ) {
              if (m.turn == Turn::CW) m.turn = Turn::CCW;
              else if (m.turn == Turn::CCW) m.turn = Turn::CW;
            } 

            moves.push_back(m);
          }
          virtualCube_->QueueFaceMoves(moves);
        }
      }
    }
    // Stay in Solve stage to display steps
  }
}

void RubicsCube::HandleReadbackResult(TensorReadbackResult&& result) {
  if (result.data.empty()) {
    Log::Write(Log::Level::Info,
               Fmt("RubicsCube|Readback: target %.*s produced an empty buffer",
                   static_cast<int>(result.name.size()), result.name.data()));
    return;
  }
  if (cameraManager_) {
    cameraManager_->OnReadbackResult(std::move(result));
  }
}

void RubicsCube::DebugFacesColor(const std::array<std::array<XrVector3f, 9>, 6>& faces) {
  constexpr int canvasSize = 200;
  constexpr int cellSize = canvasSize / 3;
  const std::filesystem::path baseDir("/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.rubics_cube/files");
  std::error_code ec;
  std::filesystem::create_directories(baseDir, ec);
  if (ec) {
    Log::Write(Log::Level::Warning,
               Fmt("RubicsCube|DebugFacesColor: failed to create %s (%s)",
                   baseDir.string().c_str(),
                   ec.message().c_str()));
  }
  auto clamp01 = [](float v) {
    return std::max(0.0f, std::min(1.0f, v));
  };
  for (int faceIdx = 0; faceIdx < 6; ++faceIdx) {
    cv::Mat canvas(canvasSize, canvasSize, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int p = 0; p < 9; ++p) {
      int row = p / 3;
      int col = p % 3;
      int x0 = col * cellSize;
      int y0 = row * cellSize;
      int x1 = (col == 2) ? canvasSize : x0 + cellSize;
      int y1 = (row == 2) ? canvasSize : y0 + cellSize;
      const XrVector3f& color = faces[faceIdx][p];
      const uint8_t r = static_cast<uint8_t>(clamp01(color.x) * 255.0f + 0.5f);
      const uint8_t g = static_cast<uint8_t>(clamp01(color.y) * 255.0f + 0.5f);
      const uint8_t b = static_cast<uint8_t>(clamp01(color.z) * 255.0f + 0.5f);
      cv::rectangle(canvas, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(b, g, r), cv::FILLED);
      cv::rectangle(canvas, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(40, 40, 40), 2);
    }
    auto filePath = baseDir / Fmt("debug_%d.png", faceIdx);
    if (cv::imwrite(filePath.string(), canvas)) {
      Log::Write(Log::Level::Info,
                 Fmt("RubicsCube|DebugFacesColor: wrote %s", filePath.string().c_str()));
    } else {
      Log::Write(Log::Level::Warning,
                 Fmt("RubicsCube|DebugFacesColor: failed to write %s", filePath.string().c_str()));
    }
  }
}

std::optional<std::string> RubicsCube::BuildFaceletsFromVirtualCube() {
  if (!virtualCube_) return std::nullopt;
  // Ensure cube pose matches initial scan pose: first face toward user, second face up.
  virtualCube_->ResetOrientation();
  ResetVirtualCubeAnchorToHead();
  std::array<std::array<XrVector3f, 9>, 6> faces{};
  for (int i = 0; i < 6; ++i) {
    auto faceColors = virtualCube_->GetFaceDisplayColors(static_cast<Face>(i));
    if (!faceColors) {
      Log::Write(Log::Level::Warning,
                 Fmt("RubicsCube|Orchestrator: missing display colors for face %d while rebuilding facelets", i));
      return std::nullopt;
    }
    faces[i] = *faceColors;
  }
  // faces order is Red, Blue, White, Orange, Green, Yellow
  // U, L, F, D, R, B
  // DebugFacesColor(faces);

  std::array<XrVector3f, 6> centers{};
  for (int i = 0; i < 6; ++i) centers[i] = faces[i][4];

  auto dist2 = [](const XrVector3f& a, const XrVector3f& b) {
    float dr = a.x - b.x;
    float dg = a.y - b.y;
    float db = a.z - b.z;
    return dr * dr + dg * dg + db * db;
  };
  const Face faceOrder[6] = {Face::U, Face::L, Face::F, Face::D, Face::R, Face::B};
  const char faceLetter[6] = {'U', 'L', 'F', 'D', 'R', 'B'};

  std::array<std::string, 6> perFace{};   // U R F D L B order
  for (auto& s : perFace) s.reserve(9);

  constexpr int orderedPatchIdx[9] = {2, 1, 0, 5, 4, 3, 8, 7, 6};       // remap 123456789 -> 321654987
  constexpr int orderedPatchIdx_Up_Down[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};  // remap 123456789 -> 321654987
  for (Face f : faceOrder) {
    int faceIdx = static_cast<int>(f);
    for (int k = 0; k < 9; ++k) {
      int p = orderedPatchIdx[k];
      if (f == Face::D || f == Face::U)  p = orderedPatchIdx_Up_Down[k];
      const XrVector3f& color = faces[faceIdx][p];
      
      // Find closed center color, index is {Red, Blue, White, Orange, Green, Yellow}
      int best = faceIdx;
      float bestDist = dist2(color, centers[faceIdx]);
      for (int c = 0; c < 6; ++c) {
        float d = dist2(color, centers[c]);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      const char ch = faceLetter[best];
      perFace[faceIdx].push_back(ch);
    }
  }

  // swap L and R order, to alias with cubeString order
  std::swap(perFace[1], perFace[4]);

  std::string out;
  out.reserve(54);
  for (const auto& s : perFace) out += s;

  for (int i = 0; i < 6; ++i) {
    Face f = faceOrder[i];
    Log::Write(Log::Level::Info,
               Fmt("RubicsCube|Orchestrator: face %s facelets=%s",
                   (f == Face::F ? "F" : f == Face::U ? "U" : f == Face::L ? "L" : f == Face::B ? "B" : f == Face::D ? "D" : "R"),
                   perFace[i].c_str()));
  }
  Log::Write(Log::Level::Info, Fmt("RubicsCube|Orchestrator: rebuilt consolidated facelets=%s", out.c_str()));
  return out;
}

bool RubicsCube::ValidateFacelets(const std::string& code) const {
  if (code.size() != 54) return false;
  std::array<int, 6> counts{};
  for (char ch : code) {
    int idx = -1;
    switch (ch) {
      case 'U': idx = 0; break;
      case 'L': idx = 1; break;
      case 'F': idx = 2; break;
      case 'D': idx = 3; break;
      case 'R': idx = 4; break;
      case 'B': idx = 5; break;
      default: return false;
    }
    counts[idx]++;
  }
  for (int c : counts) {
    if (c != 9) return false;
  }
  return true;
}

bool RubicsCube::UpdateCubeModelFromFacelets(const std::string& facelets) {
  if (!cubeModel_) {
    Log::Write(Log::Level::Warning, "VirtualCube: UpdateCubeModelFromFacelets called without cubeModel");
    return false;
  }
  if (facelets.size() != 54) {
    Log::Write(Log::Level::Warning,
               Fmt("VirtualCube: UpdateCubeModelFromFacelets received invalid length=%zu", facelets.size()));
    return false;
  }
  // Face order aligns with scanning sequence: F, U, L, B, D, R
  const Face faceOrder[6] = {Face::F, Face::U, Face::L, Face::B, Face::D, Face::R};
  size_t offset = 0;
  int unknownRaw = 0;
  for (Face f : faceOrder) {
    std::array<Color, 9> colors{};
    for (int i = 0; i < 9 && offset < facelets.size(); ++i) {
      colors[i] = LetterToColor(facelets[offset++]);
    }
    cubeModel_->SetFaceColors(f, colors);
    if (cubeModel_) {
      std::array<PatchColor, 9> raw{};
      bool haveRaw = false;
      if (virtualCube_) {
        auto display = virtualCube_->GetFaceDisplayColors(f);
        if (display) {
          for (int i = 0; i < 9; ++i) {
            raw[i] = {(*display)[i].x, (*display)[i].y, (*display)[i].z};
          }
          haveRaw = true;
        }
      }
      if (!haveRaw) {
        for (int i = 0; i < 9; ++i) raw[i] = ColorToPatch(colors[i]);
      }
      cubeModel_->SetFaceRawColors(f, raw);
      for (const auto& patch : raw) {
        if (patch.r == 0.f && patch.g == 0.f && patch.b == 0.f) {
          ++unknownRaw;
        }
      }
    }
  }
  if (virtualCube_) {
    virtualCube_->SetCubeModel(cubeModel_.get());
  }
  Log::Write(Log::Level::Info,
             Fmt("VirtualCube: applied facelets to cube model (unknownRawRGB=%d)", unknownRaw));
  return true;
}

void RubicsCube::RequestPermission(struct android_app* app) {
  Log::Write(Log::Level::Info, "RubicsCube|Orchestrator: Requesting camera permission from activity");
  RubicsCube::gapp = app;
  if (app == nullptr || app->activity == nullptr || app->activity->vm == nullptr) {
    Log::Write(Log::Level::Error, "RubicsCube|Orchestrator: Invalid android_app passed to RequestPermission");
    return;
  }

  JNIEnv* env = nullptr;
  app->activity->vm->AttachCurrentThread(&env, nullptr);
  if (env == nullptr) {
    Log::Write(Log::Level::Error, "RubicsCube|Orchestrator: Failed to attach JNI environment while requesting permission");
    return;
  }
  jobject activity = app->activity->clazz;
  jclass cls = env->GetObjectClass(activity);
  jmethodID requestCamera = env->GetMethodID(cls, "requestCameraFromNative", "()V");
  env->CallVoidMethod(activity, requestCamera);
}

bool RubicsCube::UpdateOverlayRgba(int width, int height, std::vector<uint8_t>& outRgba) {
  if (!scanUi_) return false;
  // Keep UI thread aware of overlay size
  scanUi_->SetOverlaySize(width, height);
  const bool changed = scanUi_->TryConsumeOverlayRgba(outRgba);
  if (changed) {
    Log::Write(Log::Level::Info, Fmt("RubicsCube|Overlay: composed UI RGBA (w=%d h=%d)", width, height));
  }
  return changed;
}

void RubicsCube::SetOverlaySize(int width, int height) {
  overlayW_ = width;
  overlayH_ = height;
  if (scanUi_) scanUi_->SetOverlaySize(width, height);
  Log::Write(Log::Level::Info, Fmt("RubicsCube|Orchestrator: overlay size set to %dx%d", width, height));
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<RubicsCube>(instance, session);
}

void RubicsCube::UpdateHeadPose(const XrPosef& pose) {
  lastHeadPose_ = pose;
  hasHeadPose_ = true;
}

void RubicsCube::ResetVirtualCubeAnchorToHead() {
  if (!virtualCube_ || !hasHeadPose_) return;
  const XrVector3f forwardLocal{0.f, 0.f, -1.f};
  const XrVector3f leftLocal{-1.f, 0.f, 0.f};
  XrVector3f forwardWorld{};
  XrVector3f leftWorld{};
  XrQuaternionf_RotateVector3f(&forwardWorld, &lastHeadPose_.orientation, &forwardLocal);
  XrQuaternionf_RotateVector3f(&leftWorld, &lastHeadPose_.orientation, &leftLocal);
  XrVector3f anchor = lastHeadPose_.position;
  anchor.x += forwardWorld.x * kCubeForwardOffsetM + leftWorld.x * kCubeLeftOffsetM;
  anchor.y += forwardWorld.y * kCubeForwardOffsetM + leftWorld.y * kCubeLeftOffsetM;
  anchor.z += forwardWorld.z * kCubeForwardOffsetM + leftWorld.z * kCubeLeftOffsetM;
  virtualCube_->SetAnchorPosition(anchor);
}

std::optional<Face> RubicsCube::GetFacingFaceFromHeadPose() const {
  if (!virtualCube_ || !hasHeadPose_) return std::nullopt;
  return virtualCube_->GetFacingFace(lastHeadPose_.orientation);
}

void RubicsCube::StartArrowHold(ScanUiState::Arrow arrow, const std::chrono::steady_clock::time_point& now) {
  if (arrow == ScanUiState::Arrow::None) return;
  arrowHoldActive_ = true;
  arrowHoldArrow_ = arrow;
  arrowHoldStart_ = now;
}

void RubicsCube::ClearArrowHold() {
  arrowHoldActive_ = false;
  arrowHoldArrow_ = ScanUiState::Arrow::None;
  arrowHoldStart_ = {};
}

void RubicsCube::UpdateHeadMotion(float linearDeltaM, float angularDeltaRad) {
  // Thresholds: linear 0.15m per frame or angular 0.35rad per frame
  const float linThresh = 0.10f;
  const float angThresh = 0.20f;
  if (linearDeltaM < linThresh && angularDeltaRad < angThresh) return;
  // Cooldown to avoid rapid thrash
  const auto now = std::chrono::steady_clock::now();
  if (lastMotionReset_.time_since_epoch().count() != 0) {
    if (now - lastMotionReset_ < std::chrono::milliseconds(800)) return;
  }
  lastMotionReset_ = now;

  Log::Write(Log::Level::Warning, Fmt("RubicsCube|Orchestrator: head motion detected (lin=%.3f, ang=%.3f), resetting scan", linearDeltaM, angularDeltaRad));
  // Reset scanning state
  if (cubeModel_) cubeModel_->Clear();
  scanIndex_ = 0;
  hasLastAcceptedCenter_ = false;
  solutionText_.clear();
  solutionSteps_.clear();
  cachedFacelets_.reset();
  stage_ = AppStage::Scan;
  if (virtualCube_) {
    // Clear any queued face moves so Solve animations stop
    virtualCube_->ClearFaceMoves();
  }
  if (scanManager_) scanManager_->ResetStabilizer();
  if (virtualCube_) {
    virtualCube_->SetCubeModel(cubeModel_.get());
    virtualCube_->ClearPreviewFace(scanOrder_[0]);
    virtualCube_->ClearPreview();
    virtualCube_->ResetOrientation();
    ResetVirtualCubeAnchorToHead();
  }
  awaitingValidate_ = false;
  ClearArrowHold();
}

bool RubicsCube::UpdateUserMesh(std::vector<struct Geometry::Vertex>& outVerts, std::vector<uint16_t>& outIdx,
                                XrPosef& outWorldPose) {
  if (!virtualCube_) return false;
  return virtualCube_->BuildMesh(outVerts, outIdx, outWorldPose);
}

void RubicsCube::QueueWarmupAnimation() {
  if (virtualCube_) virtualCube_->QueueWarmupAnimation();
}

}  // namespace SecureMR
