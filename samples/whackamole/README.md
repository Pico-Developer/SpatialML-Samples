## Whack‑a‑Mole (Ducks) — JavaScript Operator Collision Demo

This sample demonstrates how to integrate a JavaScript operator into a SpatialML pipeline to implement gameplay logic: detecting collisions between “moles” (duck glTFs) and human wrist joints, toggling visibility on hits, scheduling timed appearances/disappearances, and persisting a score — all driven by tensors exchanged between C++ and JavaScript.

### What This Shows

- Pose detection and landmark extraction run in SpatialML pipelines
- A JavaScript operator consumes landmark and game‑state tensors
- Per‑frame collision detection against wrist joints
- Visibility toggling and timed on/off scheduling for each duck
- Persistent score accumulation exposed to the render pipeline

### Key Files

- `samples/whackamole/cpp/whackamole_detection.cpp` — C++ pipelines, tensor wiring, asset loading
- `samples/whackamole/cpp/whackamole_detection.h` — globals, placeholders, pipeline members
- `assets/whackamole/jointsTransformer.js` — JavaScript operator logic (collisions, timers, score)

### Pipelines Overview

- Detection + Affine Update: find a person and stabilize ROI (`isPoseDetectedGlobal`, `roiAffineUpdatedGlobal`)
- Landmark: compute 4×4 transforms for bones and emit tensors to JavaScript (`bodyLandmarkGlobal`)
- Rendering: render duck instances and red boxes with per‑object visibility and poses
- Init (one‑time): compute initial human center; place ducks; seed timer deadlines

### JavaScript Operator

The operator is invoked from the Landmark pipeline and reads/writes the following tensors:

Inputs

- `positionArray` — landmark keypoints for extracting wrist positions
- `molesTransforms` — per‑duck world 4×4 transforms for rendering
- `moleCollisions` — per‑duck 4×4 transforms for collision (without stage pose)
- `realtimesincestartup` — seconds since start
- `moleOnTimings` — per‑duck “on” duration (seconds)
- `moleOffTimings` — per‑duck “off” duration (seconds)
- `moleVisibleArray` — Uint8[5], persisted visibility states
- `moleDeadlines` — Float32[5], persisted next toggle deadlines
- `score` — Int32 scalar, persisted total hits

Outputs

- `matrixArray` — bone matrices for the pose marker skeleton
- `moleVisibleArray` — updated per‑duck visibility
- `moleDeadlines` — updated deadlines after toggles and hits
- `score` — updated total

Operator Logic (high level)

- Initialize `moleVisibleArray` and `moleDeadlines` once, persist across frames
- On each frame:
  - Timers: toggle ducks on/off when `realtimesincestartup >= deadline`
  - Active ducks: compute distance to wrist joints; if within threshold, count a hit
  - On hit: increment `score`, hide duck, and set reappear deadline using `moleOnTimings`

### Timing and Persistence

- C++ regenerates `moleOnTimings` in [2, 5]s and `moleOffTimings` in [1, 2]s per frame
- A 10‑second warm‑up delays initial appearance
- `moleVisibleArray`, `moleDeadlines`, and `score` are global tensors and survive across frames

### Rendering

- Ducks: five instances, each bound to a per‑duck pose and visibility tensor
- Red boxes: one instance per body landmark matrix for wrists and ankles to visualize the landmark outputs
- TV: a glTF that displays the current `score` using `debugRenderText` onto a texture created at init

### Assets

- `Duck.gltf` — mole visual
- `redbox.gltf` — landmark visual
- `pose_marker.gltf` — skeleton driven by `bodyLandmarkGlobal`
- `scoreboard_panel.gltf` — UI surface for score

### How To Run

1. Build the sample and deploy to your target device (PICO)
2. Start the app; wait ~10s for warm‑up (ducks remain hidden)
3. Move wrists to hit visible ducks; watch score update on the TV

### Customization

- Collision radius: adjust `hitThreshold` in `jointsTransformer.js`
- Duck patterns: change offsets in the init pipeline; or drive via a tensor
- Timer ranges: modify C++ generation of `moleOnTimings` and `moleOffTimings`
- Score UI: change font, position, color in `debugRenderText`

### Notes

- Collision uses `moleCollisions` transforms (no stage pose) to avoid scene‑space bias
- Rendering uses `molesTransforms` transforms (with stage pose) to place ducks in world space
