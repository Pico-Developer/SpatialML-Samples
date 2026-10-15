import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const asset = name => new URL(`src/main/assets/yolo-det-pipeline/${name}`, import.meta.url);
const readJson = name => JSON.parse(readFileSync(asset(name), 'utf8'));
const inference = readJson('pipeline/yolo_det_inference_pipeline.json');
const labelsGraph = readJson('pipeline/yolo_det_label_pipeline.json');
const display = readJson('pipeline/yolo_det_display_pipeline.json');
const manifest = readJson('manifest.json');
const gltf = readJson('gltf/frame2_multi_10_pipeline.gltf');
const find = (graph, suffix) => graph.operators.find(op => op.type.endsWith(`_${suffix}_PICO`));
const findScript = (graph, output) => graph.operators.find(op =>
  op.type.endsWith('_JS_SCRIPTING_PICO') && op.outputs.some(ref => ref.tensor === output));
const elementCount = tensor => tensor.dimensions.reduce((a, b) => a * b, tensor.channels);
const runScript = (graph, output, values) => {
  const op = findScript(graph, output);
  assert.ok(op, `script producing ${output}`);
  const context = { ...values };
  for (const ref of op.outputs) {
    context[ref.name] = new Array(elementCount(graph.tensors[ref.tensor])).fill(0);
  }
  vm.runInNewContext(op.attrs[0], context);
  return context;
};
const close = (actual, expected, message) =>
  assert.ok(Math.abs(actual - expected) < 1e-6, `${message}: ${actual} != ${expected}`);

// Package shape and model placement match the programmatic graph.
assert.equal(manifest.schema_version, '2');
assert.deepEqual(manifest.pipelines.map(pipeline => pipeline.id), ['inference', 'labels', 'display']);
assert.equal(manifest.runtime.detection_tensor, 'detection');
const modelOps = inference.operators.filter(op => op.type.endsWith('_RUN_MODEL_INFERENCE_PICO'));
assert.equal(modelOps.length, 2);
assert.deepEqual(modelOps.map(op => op.model.model_target), ['npu', 'cpu']);
assert.equal(modelOps[0].model.bin_path, 'model/yolox_m.tflite');
assert.deepEqual(modelOps[0].model.input[0].shape, [1, 640, 640, 3]);
assert.deepEqual(modelOps[0].model.output[0].shape, [1, 8400, 85]);
assert.equal(modelOps[1].model.bin_path, 'model/yolox_postprocess.tflite');
assert.equal(modelOps[1].model.cpu_target_num_threads, 4);
assert.deepEqual(inference.tensors.candidate_scores.dimensions, [128, 1]);
assert.deepEqual(inference.tensors.candidate_boxes.dimensions, [128, 4]);
assert.deepEqual(inference.tensors.candidate_classes.dimensions, [128, 1]);
assert.deepEqual(inference.tensors.nms_scores.dimensions, [10, 1]);
assert.deepEqual(inference.tensors.nms_indices.dimensions, [10, 1]);
assert.deepEqual(inference.tensors.detection.dimensions, [10, 6]);
assert.deepEqual(inference.tensors.node_poses.dimensions, [20, 4, 4]);
assert.deepEqual(inference.outputs, ['detection', 'node_poses']);

// Class-aware NMS offsets only the coordinates submitted to NMS, preserving
// overlapping boxes from different COCO classes.
const classAware = runScript(inference, 'boxes_for_nms', {
  candidate_boxes: [10, 20, 30, 40, 10, 20, 30, 40, ...new Array(126 * 4).fill(0)],
  candidate_classes: [0, 2, ...new Array(126).fill(0)],
});
assert.deepEqual(Array.from(classAware.boxes_for_nms.slice(0, 8)), [10, 20, 30, 40, 2010, 2020, 2030, 2040]);
const nms = find(inference, 'NMS');
assert.deepEqual(nms.attrs, ['0.5']);
assert.equal(nms.inputs[1].tensor, 'boxes_for_nms');

// A detection becomes visible only after three consecutive same-class,
// IoU-matched frames. A class change at the same location starts over.
const boxes = [10, 20, 110, 220, ...new Array(127 * 4).fill(0)];
const classes = [39, ...new Array(127).fill(-1)];
const scores = [0.9, ...new Array(9).fill(0)];
const indices = [0, ...new Array(9).fill(-1)];
let state = new Array(60).fill(0);
for (let frame = 1; frame <= 3; frame++) {
  const result = runScript(inference, 'next_state', {
    candidate_boxes: boxes,
    candidate_classes: classes,
    nms_scores: scores,
    nms_indices: indices,
    state,
  });
  state = result.next_state;
  assert.equal(result.stable_scores[0], frame === 3 ? 0.9 : 0);
  assert.equal(result.stable_classes[0], 39);
  assert.equal(result.next_state[5], frame);
}
const changedClass = runScript(inference, 'next_state', {
  candidate_boxes: boxes,
  candidate_classes: [0, ...new Array(127).fill(-1)],
  nms_scores: scores,
  nms_indices: indices,
  state,
});
assert.equal(changedClass.next_state[5], 1);
assert.equal(changedClass.stable_scores[0], 0);

let clippedState = new Array(60).fill(0);
for (let frame = 1; frame <= 3; frame++) {
  const clipped = runScript(inference, 'next_state', {
    candidate_boxes: [-100, -50, 900, 800, ...new Array(127 * 4).fill(0)],
    candidate_classes: classes,
    nms_scores: scores,
    nms_indices: indices,
    state: clippedState,
  });
  clippedState = clipped.next_state;
  assert.deepEqual(Array.from(clipped.stable_boxes.slice(0, 4)), [0, 0, 639, 639]);
  assert.equal(clipped.stable_scores[0], frame === 3 ? 0.9 : 0);
}

// Camera-calibrated geometry uses the reconstructed point ray, focal lengths,
// exact decoded box size, fixed world axes, and attached horizontal labels.
const stableBoxes = new Array(40).fill(0);
const stableScores = new Array(10).fill(0);
const stableClasses = new Array(10).fill(-1);
const points = new Array(30).fill(0);
stableBoxes.splice(0, 8, 310, 315, 330, 325, 310, 315, 330, 325);
stableScores.splice(0, 2, 0.9, 0.8);
stableClasses.splice(0, 2, 0, 39);
points.splice(0, 6, 0.2, 0.1, -0.75, 0.2, 0.1, -0.75);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const geometry = runScript(inference, 'node_poses', {
  stable_boxes: stableBoxes,
  stable_scores: stableScores,
  stable_classes: stableClasses,
  point_xyz: points,
  vst_camera_matrix: [300, 0, 320, 0, 300, 320, 0, 0, 1],
  left_eye_transform: identity,
});
assert.deepEqual(Array.from(geometry.detection.slice(0, 6)), [320, 320, 20, 10, 0.9, 0]);
const frame0 = geometry.node_poses.slice(0, 16);
[[0, 0.01], [5, 0.005], [10, 0.05], [3, 0.5], [7, -0.2], [11, -1.5]]
  .forEach(([index, value]) => close(frame0[index], value, `frame0[${index}]`));
const label0 = geometry.node_poses.slice(10 * 16, 11 * 16);
[[0, 0.4], [5, 0.1], [10, 0.01], [3, 0.65], [7, -0.225], [11, -1.48]]
  .forEach(([index, value]) => close(label0[index], value, `label0[${index}]`));
const label1 = geometry.node_poses.slice(11 * 16, 12 * 16);
close(label1[7], -0.175, 'colliding label moves to box bottom');
for (const pose of [frame0, label0, label1]) {
  assert.equal(pose[1], 0);
  assert.equal(pose[4], 0);
  assert.equal(pose[6], 0);
  assert.equal(pose[9], 0);
}

const farPoints = new Array(30).fill(0);
farPoints.splice(0, 3, 10, 15, -38);
const farDepth = runScript(inference, 'node_poses', {
  stable_boxes: stableBoxes,
  stable_scores: [0.9, ...new Array(9).fill(0)],
  stable_classes: [0, ...new Array(9).fill(-1)],
  point_xyz: farPoints,
  vst_camera_matrix: [300, 0, 320, 0, 300, 320, 0, 0, 1],
  left_eye_transform: identity,
});
const farFrame = farDepth.node_poses.slice(0, 16);
close(farFrame[0], 5 * 20 / (300 * 10), 'far frame x scale');
close(farFrame[5], 5 * 10 / (300 * 10), 'far frame y scale');
close(farFrame[3], 10 * (5 / 38) + 0.1, 'far point remains on x ray');
close(farFrame[7], -(15 * (5 / 38)), 'far point remains on y ray');
close(farFrame[11], -5, 'far depth is capped');

// Stereo reconstruction can fail for a high-confidence 2D result. Such a
// result must collapse to zero scale instead of being rendered at z=0, where
// it can clip through the camera as a giant colored wedge.
const invalidDepthScores = new Array(10).fill(0);
const invalidDepthClasses = new Array(10).fill(-1);
invalidDepthScores[0] = 0.99;
invalidDepthClasses[0] = 0;
const invalidDepth = runScript(inference, 'node_poses', {
  stable_boxes: stableBoxes,
  stable_scores: invalidDepthScores,
  stable_classes: invalidDepthClasses,
  point_xyz: [Number.NaN, 0, -1, ...new Array(27).fill(0)],
  vst_camera_matrix: [300, 0, 320, 0, 300, 320, 0, 0, 1],
  left_eye_transform: identity,
});
const invalidFrame = invalidDepth.node_poses.slice(0, 16);
const invalidLabel = invalidDepth.node_poses.slice(10 * 16, 11 * 16);
for (const pose of [invalidFrame, invalidLabel]) {
  assert.equal(pose[0], 0);
  assert.equal(pose[5], 0);
  assert.equal(pose[10], 0);
  assert.ok(pose.every(Number.isFinite));
}
assert.equal(invalidFrame[11], -1.5);
assert.equal(invalidLabel[11], -1.48);

// Labels include confidence, stay blank below 50%, and retain all 80 COCO
// names in exactly the same order as the programmatic implementation.
const classSource = readFileSync(new URL('../programmatic/cpp/coco_classes.h', import.meta.url), 'utf8');
const cocoLabels = Array.from(classSource.matchAll(/"([^"]+)"/g), match => match[1]);
assert.equal(cocoLabels.length, 80);
const detections = new Array(60).fill(0);
detections.splice(0, 18,
  320, 320, 20, 10, 0.9, 0,
  320, 320, 20, 10, 0.8, 39,
  320, 320, 20, 10, 0.5, 79);
const renderedLabels = runScript(labelsGraph, 'class_text0', { detection: detections });
assert.equal(String.fromCharCode(...renderedLabels.class_text0).trimEnd(), 'person 90%');
assert.equal(String.fromCharCode(...renderedLabels.class_text1).trimEnd(), 'bottle 80%');
assert.equal(String.fromCharCode(...renderedLabels.class_text2).trimEnd(), '');
for (let index = 0; index < 10; index++) {
  assert.equal(labelsGraph.tensors[`class_text${index}`].data_type, 2);
  assert.equal(renderedLabels[`class_text${index}`].length, 18);
}
const textOps = labelsGraph.operators.filter(op => op.type.endsWith('_RENDER_TEXT_PICO'));
assert.equal(textOps.length, 10);
assert.ok(textOps.every(op => op.attrs[0] === 'bold#en-US#512#128'));
assert.deepEqual(textOps.map(op => op.inputs[4].tensor),
  Array.from({ length: 10 }, (_, index) => `texture_id${index}`));

// Both render stages share one 20-node glTF. This avoids the Vulkan descriptor
// pressure seen with one glTF per detection while keeping per-slot colors.
assert.equal(labelsGraph.tensors.detection_gltf.asset, 'gltf/frame2_multi_10_pipeline.gltf');
assert.equal(display.tensors.detection_gltf.asset, 'gltf/frame2_multi_10_pipeline.gltf');
assert.deepEqual(display.inputs, ['node_poses']);
assert.deepEqual(display.tensors.node_ids.data, Array.from({ length: 20 }, (_, index) => index));
assert.equal(display.operators.filter(op => op.type.endsWith('_UPDATE_GLTF_PICO')).length, 1);
assert.equal(display.operators.filter(op => op.type.endsWith('_SWITCH_GLTF_RENDER_STATUS_PICO')).length, 1);
assert.equal(gltf.nodes.length, 20);
assert.deepEqual(gltf.scenes[0].nodes, Array.from({ length: 20 }, (_, index) => index));
assert.ok(gltf.nodes.every(node =>
  Array.isArray(node.scale) && node.scale.length === 3 && node.scale.every(value => value === 0)));
assert.equal(gltf.meshes.length, 20);
assert.equal(gltf.materials.length, 20);
assert.equal(gltf.textures.length, 10);
assert.equal(gltf.images.length, 10);

// Model, camera, and stereo bindings remain end-to-end connected.
const camera = find(inference, 'RECTIFIED_VST_ACCESS');
const imageCopy = inference.operators.find(op => op.outputs.some(ref => ref?.tensor === 'image_f32'));
assert.equal(imageCopy.inputs[0].tensor, camera.outputs[0].tensor);
assert.equal(imageCopy.type, 'XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO');
const uv = find(inference, 'UV_TO_3D_IN_CAM_SPACE');
assert.equal(uv.inputs[1].tensor, camera.outputs[2].tensor);
assert.equal(uv.inputs[3].tensor, camera.outputs[0].tensor);
assert.equal(uv.inputs[4].tensor, camera.outputs[1].tensor);
assert.equal(find(inference, 'CAMERA_SPACE_TO_WORLD').inputs[0].tensor, camera.outputs[2].tensor);

// The packaged copies must remain byte-identical to the programmatic assets.
const sha256 = url => createHash('sha256').update(readFileSync(url)).digest('hex');
assert.equal(sha256(asset('model/yolox_postprocess.tflite')),
  sha256(new URL('../../../assets/yolo_det_programmatic/yolox_postprocess.tflite', import.meta.url)));
const programmaticGltf = JSON.parse(readFileSync(
  new URL('../../../assets/yolo_det_programmatic/frame2_multi_10.gltf', import.meta.url), 'utf8'));
assert.deepEqual(gltf.meshes, programmaticGltf.meshes);
assert.deepEqual(gltf.materials, programmaticGltf.materials);
assert.deepEqual(gltf.buffers, programmaticGltf.buffers);

console.log('YOLO pipeline parity, geometry, stabilization, label, and asset checks passed.');
