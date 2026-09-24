#!/bin/bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 [--input input_tensor.bin] [--duration seconds] [--output expected_values] [--output-name src_output_file_name] --model <path_to_bin> --json <path_to_json>
  --input       Input tensor file (NHWC float32). Optional; random input is used if omitted.
  --duration    Run time in seconds (default: 20)
  --output      Expected output: comma-separated float32 values or a float32 binary file (first output tensor)
  --output-name Model output file name to compared with --output
  --model       Serialized model .bin
  --json        Model spec .json
Examples:
  ./scripts/run_model_inspect.sh --model assets/mnistwild/mnist.serialized.bin --json assets/mnistwild/mnist.serialized.json
  ./scripts/run_model_inspect.sh --model assets/pose/detection.serialized.bin --json assets/pose/detection.serialized.json
  ./scripts/run_model_inspect.sh --model assets/pose/landmark.serialized.bin --json assets/pose/landmark.serialized.json
  ./scripts/run_model_inspect.sh --model assets/UFO/facedetector_fp16_qnn229.bin --json assets/UFO/facedetector_fp16_qnn229.json
  ./scripts/run_model_inspect.sh --model assets/yolo_det/yolo.serialized.bin --json assets/yolo_det/yolo.serialized.json
  ./scripts/run_model_inspect.sh --model assets/yolo_det/yolom.serialized.bin --json assets/yolo_det/yolom.serialized.json
EOF
    exit 1
}

INPUT_FILE=""
RUNTIME=20
EXPECTED_VALUES=""
MODEL_FILE=""
JSON_FILE=""
OUTPUT_FILE=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --duration)
            RUNTIME="$2"
            shift 2
            ;;
        --output)
            EXPECTED_VALUES="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --json)
            JSON_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        -*)
            usage
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -ne 0 ]; then
    usage
fi

if [ -z "$MODEL_FILE" ] || [ -z "$JSON_FILE" ]; then
    usage
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Bin file not found: $MODEL_FILE"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "Error: Json file not found: $JSON_FILE"
    exit 1
fi

if [ -n "$INPUT_FILE" ] && [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input tensor file not found: $INPUT_FILE"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

PACKAGE_NAME="com.bytedance.pico.spatial_ml_demo.model_inspect"
COMPONENT="$PACKAGE_NAME/android.app.NativeActivity"
# Keep under Android setprop length limits; use app external storage.
DEVICE_TMP_DIR="/sdcard/Android/data/${PACKAGE_NAME}/files"
DEVICE_INPUT_PATH="$DEVICE_TMP_DIR/input.bin"
DEVICE_OUTPUT_DIRS=("${DEVICE_TMP_DIR}/model_inspect" "/data/local/tmp/securemr_model_inspect")
LOCAL_OUTPUT_DIR="${REPO_ROOT}/tmp_data/model_inspect_outputs_$(date +%Y%m%d_%H%M%S)"
LOGCAT_PID=""

screen_is_on() {
    local state
    state="$(adb shell dumpsys power 2>/dev/null | grep -m1 -E 'Display Power|mWakefulness' || true)"
    if echo "$state" | grep -q -E 'ON|Awake'; then
        return 0
    fi
    return 1
}

ensure_screen_on() {
    if screen_is_on; then
        return
    fi
    echo "Waking device screen..."
    adb shell input keyevent 26
    sleep 3
}

turn_screen_off() {
    if screen_is_on; then
        adb shell input keyevent 26 >/dev/null 2>&1 || true
    fi
}

cleanup() {
    if [ -n "$LOGCAT_PID" ]; then
        kill "$LOGCAT_PID" 2>/dev/null || true
        LOGCAT_PID=""
    fi
    adb shell am force-stop "$PACKAGE_NAME" >/dev/null 2>&1 || true
    turn_screen_off
}
trap cleanup EXIT

echo "Building and installing model_inspect..."
cd "$REPO_ROOT"
./gradlew :samples:model_inspect:installDebug

echo "Cleaning device temp directory: $DEVICE_TMP_DIR"
adb shell rm -rf "$DEVICE_TMP_DIR"
adb shell mkdir -p "$DEVICE_TMP_DIR"

echo "Cleaning output directories: ${DEVICE_OUTPUT_DIRS[*]}"
for dir in "${DEVICE_OUTPUT_DIRS[@]}"; do
    adb shell rm -f "$dir"/* >/dev/null 2>&1 || true
done

echo "--------------------------------------------------"
echo "Testing model: $(basename "$MODEL_FILE")"

echo "Pushing model files to $DEVICE_TMP_DIR..."
adb push "$MODEL_FILE" "$DEVICE_TMP_DIR/model.serialized.bin" >/dev/null
adb push "$JSON_FILE" "$DEVICE_TMP_DIR/model.serialized.json" >/dev/null

if [ -n "$INPUT_FILE" ]; then
    echo "Pushing input tensor to $DEVICE_INPUT_PATH (NHWC float32 assumed)..."
    adb push "$INPUT_FILE" "$DEVICE_INPUT_PATH" >/dev/null
    adb shell setprop debug.securemr.model_inspect.input "$DEVICE_INPUT_PATH"
else
    adb shell setprop debug.securemr.model_inspect.input "''"
fi

echo "Setting property debug.securemr.model_inspect.model_dir to $DEVICE_TMP_DIR"
adb shell setprop debug.securemr.model_inspect.model_dir "$DEVICE_TMP_DIR"

ensure_screen_on

existing_pid="$(adb shell pidof "$PACKAGE_NAME" 2>/dev/null | tr -d '\r' || true)"
if [ -n "$existing_pid" ]; then
    echo "App already running (pid $existing_pid); stopping before relaunch..."
    adb shell am force-stop "$PACKAGE_NAME"
    sleep 3
fi

echo "Starting logcat..."
export PYTHONUNBUFFERED=1
"$REPO_ROOT/scripts/logcat" -p "$PACKAGE_NAME" -e ModelInspect 2>&1 &
LOGCAT_PID=$!

echo "Launching app..."
adb shell am force-stop com.bytedance.pico.openmr
sleep 2
adb shell am start -n "$COMPONENT"

echo "Waiting for ${RUNTIME} seconds..."
sleep "$RUNTIME"

echo "Stopping app..."
adb shell am force-stop "$PACKAGE_NAME"

mkdir -p "$LOCAL_OUTPUT_DIR"
echo "Pulling outputs to $LOCAL_OUTPUT_DIR..."
outputs_pulled=0
for dir in "${DEVICE_OUTPUT_DIRS[@]}"; do
    mapfile -t files < <(adb shell ls "$dir" 2>/dev/null | tr -d '\r' || true)
    if [ "${#files[@]}" -eq 0 ]; then
        continue
    fi
    for f in "${files[@]}"; do
        if [[ "$f" == model_inspect_output_*.bin ]]; then
            adb shell cat "$dir/$f" >"$LOCAL_OUTPUT_DIR/$f"
            outputs_pulled=1
        fi
    done
done

if [ "$outputs_pulled" -eq 1 ]; then
    echo "Outputs saved under $LOCAL_OUTPUT_DIR"
else
    echo "No output files pulled (none found on device)."
fi

if [ -n "$EXPECTED_VALUES" ] && [ "$outputs_pulled" -eq 1 ]; then
    echo "$REPO_ROOT/scripts/compare_output.py" "$EXPECTED_VALUES" "$LOCAL_OUTPUT_DIR" "$OUTPUT_FILE"
    python3 "$REPO_ROOT/scripts/compare_output.py" "$EXPECTED_VALUES" "$LOCAL_OUTPUT_DIR" "$OUTPUT_FILE"
fi

echo "Turning off screen..."
turn_screen_off

echo "--------------------------------------------------"
echo "Test completed."
