# Whack-a-Mole Pipeline

Schema-v2 package counterpart to the [programmatic game](../programmatic/README.md). The package under [`src/main/assets/whackamole-pipeline`](src/main/assets/whackamole-pipeline) bundles LiteRT NPU pose detection and landmark models, game/display JSON, and the duck, pose marker, red box, and scoreboard glTF assets.

This sample uses a dedicated native host instead of `pipeline_common`. The host supplies `realtime_since_startup`, submits `scoreboard_init` once, and then chains `inference` and `display` every frame. Collision, visibility, deadlines, and scoring remain package-local JavaScript/tensor logic. No tensor readback is required for gameplay.

```sh
./gradlew :samples:whackamole_pipeline:installDebug
```

Launch **Whack-a-Mole Pipeline**, wait for the initial game delay, and move a wrist through visible ducks. The scoreboard should increment and ducks should follow their appearance timers.
