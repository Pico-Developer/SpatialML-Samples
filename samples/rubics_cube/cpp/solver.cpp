// Copyright (2025) Bytedance Ltd. and/or its affiliates
// Licensed under the Apache License, Version 2.0

#include "solver.h"
#include "logger.h"
#include "common.h"

extern "C" {
#include "search.h"
}

#if defined(XR_USE_PLATFORM_ANDROID)
// Provided by base/main.cpp on Android
extern std::string g_internalDataPath;
#endif

namespace SecureMR {

std::string Solver::Solve(const std::string& code) {
  Log::Write(Log::Level::Info, Fmt("RubicsCube|Solver: received facelets len=%zu", code.size()));
  if (code.size() != 54) {
    Log::Write(Log::Level::Error, "RubicsCube|Solver: invalid facelets length");
    return std::string();
  }

  // Prepare inputs
  std::string facelets = code;  // already URFDLB letters
  int maxDepth = 21;
  long timeoutSec = 5;  // safety timeout
  int useSeparator = 0; // do not insert '.'
#if defined(XR_USE_PLATFORM_ANDROID)
  std::string cache = g_internalDataPath.empty() ? std::string("/sdcard/Android/data/com.bytedance.pico.spatial_ml_demo.rubics_cube/cache/kociemba")
                                                 : (g_internalDataPath + std::string("/kociemba"));
#else
  std::string cache = std::string("cache/kociemba");
#endif

  char* cstr = solution(facelets.data(), maxDepth, timeoutSec, useSeparator, cache.c_str());
  if (!cstr) {
    Log::Write(Log::Level::Warning, "RubicsCube|Solver: no solution returned (invalid cube or timeout)");
    return std::string();
  }
  std::string out(cstr);
  free(cstr);
  return out;
}

}  // namespace SecureMR
