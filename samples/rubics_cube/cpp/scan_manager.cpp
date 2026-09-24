// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "scan_manager.h"
#include "logger.h"
#include "common.h"

#include <filesystem>
#include <cmath>
#include <algorithm>

#if defined(XR_USE_PLATFORM_ANDROID)
// OpenCV for debugging image dump
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
// Provided by base/main.cpp on Android
extern std::string g_internalDataPath;
#endif

namespace SecureMR {

static PatchColor AverageRegion(const Frame& f, int x0, int y0, int x1, int y1) {
  // Clamp to frame bounds
  x0 = std::max(0, x0);
  y0 = std::max(0, y0);
  x1 = std::min(f.width, x1);
  y1 = std::min(f.height, y1);
  if (x1 <= x0 || y1 <= y0 || f.channels < 3 || f.mat.empty()) return {};

  // Quantized color histogram (mode). Use 16 bins per channel.
  constexpr int kBins = 16;
  constexpr int kBins2 = kBins * kBins;
  constexpr int kBins3 = kBins * kBins * kBins;
  std::array<int, kBins3> counts{};
  std::array<double, kBins3> sumR{};
  std::array<double, kBins3> sumG{};
  std::array<double, kBins3> sumB{};

  for (int y = y0; y < y1; ++y) {
    const uint8_t* row = f.mat.ptr<uint8_t>(y);
    for (int x = x0; x < x1; ++x) {
      const uint8_t* px = row + x * f.channels;  // BGR
      const int b = px[0];
      const int g = px[1];
      const int r = px[2];
      int rb = std::min(kBins - 1, (r * kBins) >> 8);
      int gb = std::min(kBins - 1, (g * kBins) >> 8);
      int bb = std::min(kBins - 1, (b * kBins) >> 8);
      const int idx = (rb * kBins2) + (gb * kBins) + bb;
      counts[idx] += 1;
      sumR[idx] += r;
      sumG[idx] += g;
      sumB[idx] += b;
    }
  }

  // Find the dominant bin
  int bestIdx = -1;
  int bestCount = 0;
  for (int i = 0; i < kBins3; ++i) {
    if (counts[i] > bestCount) {
      bestCount = counts[i];
      bestIdx = i;
    }
  }
  if (bestIdx < 0 || bestCount == 0) return {};

  const float r = static_cast<float>(sumR[bestIdx] / bestCount) / 255.0f;
  const float g = static_cast<float>(sumG[bestIdx] / bestCount) / 255.0f;
  const float b = static_cast<float>(sumB[bestIdx] / bestCount) / 255.0f;
  return PatchColor{r, g, b};
}

bool ScanManager::TryLockFace(Face target, const Frame& frame) {
  if (frame.width <= 0 || frame.height <= 0 || frame.channels < 3 || frame.mat.empty()) return false;

  const auto now = std::chrono::steady_clock::now();
  // Global pause to avoid cross-face crosstalk after locking
  if (pauseUntil_.time_since_epoch().count() != 0 && now < pauseUntil_) {
    return false;
  }
  static const auto kInterval = std::chrono::milliseconds(200);
  if (lastSampleTime_.time_since_epoch().count() != 0) {
    const auto since = now - lastSampleTime_;
    if (since < kInterval) {
      // Not time yet; just keep UI color as last and continue
      return false;
    }
  }
  const float dtSec = (lastSampleTime_.time_since_epoch().count() == 0)
                          ? 0.f
                          : std::chrono::duration_cast<std::chrono::duration<float>>(now - lastSampleTime_).count();
  lastSampleTime_ = now;

  Log::Write(Log::Level::Debug,
             Fmt("RubicsCube|ScanManager: TryLockFace(%d) on frame %dx%dx%d", static_cast<int>(target), frame.width,
                 frame.height, frame.channels));

  // Simple grid, alias with XrCompositionLayerQuad rectangle
  XrRect2Di rect = GetRoiRectPx(frame.width, frame.height);
  const int box_x = rect.extent.width * 0.8 - 25;
  const int box_y = rect.extent.width * 0.8;
  const int x0 = rect.offset.x + 55 + 10;
  const int y0 = rect.offset.y - 7 + 5;

  const int step_x = box_x / 3;
  const int step_y = box_y / 3;

  // Dump the central box region to cache for debugging (Android only)
#if defined(XR_USE_PLATFORM_ANDROID)
  try {
    const int rx0 = std::max(0, x0);
    const int ry0 = std::max(0, y0);
    const int rw = std::min(box_x, frame.width - rx0);
    const int rh = std::min(box_y, frame.height - ry0);

    if (rw > 0 && rh > 0 && !frame.mat.empty()) {
      cv::Rect roiRect(rx0, ry0, rw, rh);
      cv::Mat bgr = frame.mat(roiRect);
    
      // Property-gated image saving
      namespace fs = std::filesystem;
      fs::path cacheDir;
      if (!g_internalDataPath.empty()) {
        cacheDir = fs::path(g_internalDataPath) / "cache";
      } else {
        cacheDir = fs::path("/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.rubics_cube/cache");
      }
      std::error_code ec;
      fs::create_directories(cacheDir, ec);
      char prop[PROP_VALUE_MAX] = {};
      if (__system_property_get("rubicscube_debug_save_image", prop) != 0 && prop[0] == '1') {
        const auto now = std::chrono::system_clock::now();
        const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        const std::string fileName =
            Fmt("rc_box_face%d_%dx%dx%d_%lld.png", static_cast<int>(target), rw, rh, frame.channels, (long long)ts);
        const fs::path filePath = cacheDir / fileName;


        const bool ok = cv::imwrite(filePath.string(), bgr);
        Log::Write(Log::Level::Info,
                   Fmt("RubicsCube|ScanManager: box ROI %dx%dx%d saved=%s path=%s", rw, rh, frame.channels,
                       ok ? "true" : "false", filePath.string().c_str()));
      }
    } else {
      Log::Write(Log::Level::Warning, "RubicsCube|ScanManager: ROI not saved due to invalid size/channels");
    }
  } catch (const std::exception& ex) {
    Log::Write(Log::Level::Error, Fmt("RubicsCube|ScanManager: ROI save exception: %s", ex.what()));
  }
#endif

  ScannedFace face{};
  face.face = target;
  int idx = 0;
  for (int gy = 0; gy < 3; ++gy) {
    for (int gx = 0; gx < 3; ++gx) {
      const int rx0 = x0 + gx * step_x + step_x / 6;
      const int ry0 = y0 + gy * step_y + step_y / 6;
      const int rx1 = rx0 + step_x * 2 / 3;
      const int ry1 = ry0 + step_y * 2 / 3;
      face.patches[idx++] = AverageRegion(frame, rx0, ry0, rx1, ry1);
    }
  }

  // Calibrate center patch to the face for color prototypes
  if (classifier_) {
    classifier_->CalibrateCenter(target, face.patches[4]);
  }

  // Convert patches to 8-bit for comparison/UI
  auto clampU8 = [](int v) { return static_cast<uint8_t>(std::min(std::max(v, 0), 255)); };
  std::array<std::array<uint8_t, 3>, 9> currentColors{};
  for (int i = 0; i < 9; ++i) {
    const auto& pc = face.patches[i];
    currentColors[i][0] = clampU8(static_cast<int>(std::round(pc.r * 255.0f)));
    currentColors[i][1] = clampU8(static_cast<int>(std::round(pc.g * 255.0f)));
    currentColors[i][2] = clampU8(static_cast<int>(std::round(pc.b * 255.0f)));
  }

  // Tolerant compare to mitigate sensor noise
  bool sameAsPrev = hasPrev_;
  const int kTol = 20;  // per-channel tolerance in 8-bit
  if (hasPrev_) {
    for (int i = 0; i < 9 && sameAsPrev; ++i) {
      const int dr = std::abs((int)currentColors[i][0] - (int)prevPatchColors9U8_[i][0]);
      const int dg = std::abs((int)currentColors[i][1] - (int)prevPatchColors9U8_[i][1]);
      const int db = std::abs((int)currentColors[i][2] - (int)prevPatchColors9U8_[i][2]);
      if (dr > kTol || dg > kTol || db > kTol) {
        sameAsPrev = false;
      }
    }
  }
  const int targetSamples = std::max(1, static_cast<int>(std::round(stableThresholdSec_ / 0.2f))); // ~5 for 1s/200ms
  if (!hasPrev_ || !sameAsPrev) {
    prevPatchColors9U8_ = currentColors;
    hasPrev_ = true;
    stableAccumSec_ = 0.f;
    sampleHistory_.clear();
    sampleHistory_.push_back(currentColors);
    sampleSeq_ = 1;
  } else {
    stableAccumSec_ += dtSec;
    if (sampleHistory_.empty() || sampleHistory_.back() != currentColors) {
      sampleHistory_.push_back(currentColors);
      if (static_cast<int>(sampleHistory_.size()) > targetSamples) {
        // Keep bounded history
        sampleHistory_.erase(sampleHistory_.begin());
      }
    }
    sampleSeq_++;
  }

  // Debug: if current face seems stable at this 200ms sample, save ROI to cache (Android only)
#if defined(XR_USE_PLATFORM_ANDROID)
  try {
    char prop_face[PROP_VALUE_MAX] = {};
    if (__system_property_get("rubicscube_debug_save_face", prop_face) != 0 && prop_face[0] == '1') {
      if (sameAsPrev) {
        const int rx0 = std::max(0, x0);
        const int ry0 = std::max(0, y0);
        const int rw = std::min(box_x, frame.width - rx0);
        const int rh = std::min(box_y, frame.height - ry0);
        if (rw > 0 && rh > 0 && !frame.mat.empty()) {
          namespace fs = std::filesystem;
          fs::path cacheDir;
          if (!g_internalDataPath.empty()) {
            cacheDir = fs::path(g_internalDataPath) / "cache";
          } else {
            cacheDir = fs::path("/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.rubics_cube/cache");
          }
          std::error_code ec;
          fs::create_directories(cacheDir, ec);
          cv::Rect roiRect(rx0, ry0, rw, rh);
          cv::Mat roi = frame.mat(roiRect);
          // Build 9-char color code using currentColors classification
          std::string colorCode;
          colorCode.reserve(9);
          for (int i = 0; i < 9; ++i) {
            PatchColor pc{ currentColors[i][0] / 255.0f, currentColors[i][1] / 255.0f, currentColors[i][2] / 255.0f };
            Color lbl = classifier_ ? classifier_->Classify(pc) : Color::Unknown;
            std::string ch = ColorToChar(lbl);
            colorCode += ch.empty() ? '?' : ch[0];
          }
          const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
          const fs::path filePath = cacheDir /
                                     Fmt("rc_face_sample_face%d_seq%03d_%s_%dx%dx%d_%lld.png",
                                         static_cast<int>(target), sampleSeq_, colorCode.c_str(), rw, rh, frame.channels,
                                         (long long)ts);
          const bool ok = cv::imwrite(filePath.string(), roi);
          Log::Write(Log::Level::Debug,
                     Fmt("RubicsCube|ScanManager: stable face ROI %dx%dx%d saved=%s path=%s (seq=%d code=%s)", rw, rh,
                         frame.channels, ok ? "true" : "false", filePath.string().c_str(), sampleSeq_, colorCode.c_str()));
        }
      }
    }
  } catch (const std::exception& ex) {
    Log::Write(Log::Level::Error, Fmt("RubicsCube|ScanManager: stable face save exception: %s", ex.what()));
  }
#endif

  // Update UI colors regardless
  lastPatchColors9U8_ = currentColors;
  const auto& c = face.patches[4];
  const int r8 = static_cast<int>(std::round(c.r * 255.0f));
  const int g8 = static_cast<int>(std::round(c.g * 255.0f));
  const int b8 = static_cast<int>(std::round(c.b * 255.0f));
  lastCenterColorU8_[0] = clampU8(r8);
  lastCenterColorU8_[1] = clampU8(g8);
  lastCenterColorU8_[2] = clampU8(b8);

  // If stable for threshold, filter samples by majority vote (class labels), then average inliers
  if (stableAccumSec_ >= stableThresholdSec_) {
    std::array<std::array<uint8_t, 3>, 9> filtered = currentColors;
    if (!sampleHistory_.empty()) {
      auto colorToIndex = [](Color c) -> int {
        switch (c) {
          case Color::W: return 0;
          case Color::Y: return 1;
          case Color::R: return 2;
          case Color::O: return 3;
          case Color::G: return 4;
          case Color::B: return 5;
          default: return 6;  // Unknown
        }
      };

      for (int i = 0; i < 9; ++i) {
        int counts[7] = {0};
        std::vector<int> labels; labels.reserve(sampleHistory_.size());
        for (const auto& s : sampleHistory_) {
          PatchColor pc{ s[i][0] / 255.0f, s[i][1] / 255.0f, s[i][2] / 255.0f };
          Color lbl = classifier_ ? classifier_->Classify(pc) : Color::Unknown;
          int idx = colorToIndex(lbl);
          labels.push_back(idx);
          counts[idx]++;
        }
        int bestIdx = 6; int bestCnt = -1;
        for (int k = 0; k < 7; ++k) { if (counts[k] > bestCnt) { bestCnt = counts[k]; bestIdx = k; } }
        // Average inliers with the majority label; fallback to last sample if all Unknown
        int sr = 0, sg = 0, sb = 0, n = 0;
        for (size_t si = 0; si < sampleHistory_.size(); ++si) {
          if (labels[si] == bestIdx && bestIdx != 6) {
            sr += sampleHistory_[si][i][0];
            sg += sampleHistory_[si][i][1];
            sb += sampleHistory_[si][i][2];
            n++;
          }
        }
        if (n > 0) {
          filtered[i][0] = static_cast<uint8_t>(sr / n);
          filtered[i][1] = static_cast<uint8_t>(sg / n);
          filtered[i][2] = static_cast<uint8_t>(sb / n);
        } else {
          // Fallback: latest sample
          const auto& s = sampleHistory_.back();
          filtered[i][0] = s[i][0];
          filtered[i][1] = s[i][1];
          filtered[i][2] = s[i][2];
        }
      }
    }
    // Update UI colors with filtered result
    lastPatchColors9U8_ = filtered;
    lastCenterColorU8_ = filtered[4];

    // Build final locked face from filtered colors (convert to 0..1 floats)
    ScannedFace finalFace = face;
    for (int i = 0; i < 9; ++i) {
      finalFace.patches[i].r = filtered[i][0] / 255.0f;
      finalFace.patches[i].g = filtered[i][1] / 255.0f;
      finalFace.patches[i].b = filtered[i][2] / 255.0f;
    }
    if (classifier_) {
      classifier_->CalibrateCenter(target, finalFace.patches[4]);
    }
    locked_ = finalFace;
    Log::Write(Log::Level::Info,
               Fmt("RubicsCube|ScanManager: locked face %d after %.2fs", static_cast<int>(target), stableAccumSec_));
    // Reset for next face
    hasPrev_ = false;
    stableAccumSec_ = 0.f;
    sampleHistory_.clear();
    sampleSeq_ = 0;
    return true;
  }
  return false;
}

std::optional<ScannedFace> ScanManager::GetLockedFace() { return locked_; }

XrRect2Di ScanManager::GetRoiRectPx(int W, int H) {
  // assert W = H = 1024
  const int rectW = 200;
  const int rectH = 200;
  const int left = static_cast<int>(std::round(W / 2.0f));
  const int top = static_cast<int>(std::round(H / 2.0f));

  XrRect2Di r{};
  r.offset.x = left;
  r.offset.y = top;
  r.extent.width = rectW;
  r.extent.height = rectH;
  return r;
}

void ScanManager::ResetStabilizer() {
  hasPrev_ = false;
  stableAccumSec_ = 0.f;
  sampleHistory_.clear();
  sampleSeq_ = 0;
}

}  // namespace SecureMR
