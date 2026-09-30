import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const read = name => JSON.parse(readFileSync(new URL(`src/main/assets/yolo-det-pipeline/pipeline/${name}.json`, import.meta.url)));
const inference = read('yolo_det_inference_pipeline');
const display = read('yolo_det_display_pipeline');
const find = (graph, suffix) => graph.operators.find(op => op.type.endsWith(`_${suffix}_PICO`));
const runScript = (graph, output, values) => {
  const op = graph.operators.find(op => op.type.endsWith('_JS_SCRIPTING_PICO') && op.outputs.some(ref => ref.tensor === output));
  const context = { ...values };
  for (const ref of op.outputs) {
    const tensor = graph.tensors[ref.tensor];
    context[ref.name] = new Array(tensor.dimensions.reduce((a, b) => a * b, tensor.channels)).fill(0);
  }
  vm.runInNewContext(op.attrs[0], context);
  return context;
};

// Regressions: a small box must not be enlarged by the old minimum scale,
// and camera points nearer than 1.5 m must move along the same ray.
for (const [point, size, translation, scale] of [
  [[0.2, 0.1, -0.75], [20, 10], [0.5, -0.2, -1.5], [0.01, 0.005, 0.05]],
  [[-0.4, -0.2, -3], [400, 200], [-0.3, 0.2, -3], [0.2, 0.1, 0.05]],
  [[0, 0, -1.5], [640, 640], [0.1, 0, -1.5], [0.32, 0.32, 0.05]],
]) {
  const actual = runScript(inference, 'frame_translation', { detection: [320, 320, ...size, 0.9, 0], point_xyz: point });
  for (const [key, expected] of [['frame_translation', translation], ['frame_scale', scale]]) {
    expected.forEach((value, i) => assert.ok(Math.abs(actual[key][i] - value) < 1e-6, `${key}[${i}]`));
  }
}

// Preserve world translation and diagonal scale while matching the existing
// programmatic renderer's removal of the four Y cross-axis entries.
const world = [1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30, 0, 0, 0, 1];
const pose = runScript(inference, 'frame_pose', { world_pose: world }).frame_pose;
assert.deepEqual(Array.from(pose), [1, 0, 3, 10, 0, 5, 0, 20, 7, 0, 9, 30, 0, 0, 0, 1]);
for (const [score, visible] of [[0.2, 0], [0.6, 0], [0.61, 1]]) {
  const result = runScript(display, 'frame_visible', { detection: [320, 320, 100, 100, score, 0] });
  assert.equal(result.frame_visible[0], visible);
}

// Class IDs must use the same COCO ordering as the programmatic sample. Check
// all 80 labels, including names longer than its historical 13-byte buffer.
const classSource = readFileSync(new URL('../programmatic/cpp/coco_classes.h', import.meta.url), 'utf8');
const labels = Array.from(classSource.matchAll(/"([^"]+)"/g), match => match[1]);
assert.equal(labels.length, 80);
for (const [classId, label] of [...labels.map((label, i) => [i, label]), [-1, 'unknown'], [80, 'unknown'], [0.5, 'unknown']]) {
  const result = runScript(display, 'class_text', { detection: [320, 320, 100, 100, 0.9, classId] });
  assert.equal(String.fromCharCode(...result.class_text).trimEnd(), label);
  assert.equal(result.class_text.length, 16);
  assert.ok(result.class_text.every(value => value >= 32 && value < 127));
}
assert.equal(display.tensors.class_text.data_type, 2); // INT8 renders as text.
assert.equal(find(display, 'RENDER_TEXT').inputs[0].tensor, 'class_text');

// End-to-end wiring: use the same camera output as programmatic, reconstruct
// depth from that frame, and pass its computed world pose to the display.
const camera = find(inference, 'RECTIFIED_VST_ACCESS');
const imageCopy = inference.operators.find(op => op.outputs.some(ref => ref?.tensor === 'image_f32'));
assert.equal(imageCopy.inputs[0].tensor, camera.outputs[0].tensor);
const uv = find(inference, 'UV_TO_3D_IN_CAM_SPACE');
assert.equal(uv.inputs[3].tensor, camera.outputs[0].tensor);
assert.equal(uv.inputs[4].tensor, camera.outputs[1].tensor);
assert.equal(uv.inputs[1].tensor, camera.outputs[2].tensor);
const cameraWorld = find(inference, 'CAMERA_SPACE_TO_WORLD');
const multiply = inference.operators.find(op => op.outputs.some(ref => ref?.tensor === 'world_pose'));
assert.equal(multiply.inputs[0].tensor, cameraWorld.outputs[1].tensor);
assert.equal(cameraWorld.inputs[0].tensor, camera.outputs[2].tensor);
assert.ok(inference.outputs.includes('frame_pose'));
assert.ok(display.inputs.includes('frame_pose'));
assert.equal(find(display, 'SWITCH_GLTF_RENDER_STATUS').inputs[1].tensor, 'frame_pose');
assert.ok(!display.operators.some(op => op.outputs.some(ref => ref?.tensor === 'frame_pose')));
assert.equal(find(display, 'RENDER_TEXT').attrs[0], 'sans_serif#en-us#1024#1024');
console.log('YOLO pipeline geometry, camera binding, and COCO label checks passed.');
