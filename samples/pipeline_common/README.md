# Pipeline package sample host

`pipeline_package_app.cpp` is the shared native host for straightforward loader-driven samples. Each consuming Android module supplies these CMake definitions:

- `PIPELINE_PACKAGE_ROOT` — asset directory containing `manifest.json`
- `PIPELINE_IDS` — comma-separated submission order
- `PIPELINE_APP_TAG` — log prefix
- `PIPELINE_FRAME_WIDTH` and `PIPELINE_FRAME_HEIGHT` — SpatialML camera framework size

The host loads the schema-v2 package with `SecureMrUtils::LoadModelPackagePipelinesFromAssets(...)`, resolves the configured pipeline IDs, and chains each submission to the preceding run.

Current consumers are Pose, YOLO Detection, and MNIST Wild pipeline modules. Whack-a-Mole and Stylize use dedicated hosts because they require sample-specific timing, controller input, and per-side tensor state. Face also uses a small dedicated host for its fixed camera configuration.
