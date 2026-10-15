#!/usr/bin/env node

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const pipelineDir = resolve(root, 'src/main/assets/yolo-det-pipeline/pipeline');
const manifestPath = resolve(root, 'src/main/assets/yolo-det-pipeline/manifest.json');
const programmaticGltfPath = resolve(root, '../../../assets/yolo_det_programmatic/frame2_multi_10.gltf');
const pipelineGltfPath = resolve(root, 'src/main/assets/yolo-det-pipeline/gltf/frame2_multi_10_pipeline.gltf');
const classHeader = readFileSync(resolve(root, '../programmatic/cpp/coco_classes.h'), 'utf8');
const labels = Array.from(classHeader.matchAll(/"([^"]+)"/g), match => match[1]);

const COUNT = 10;
const CANDIDATES = 128;
const THRESHOLD = 0.5;
const CONFIRMATION_FRAMES = 3;
const CONFIRMATION_IOU = 0.4;
const LABEL_LENGTH = 18;
const COLORS = [
  [255, 90, 95, 255], [0, 212, 255, 255], [124, 252, 0, 255], [255, 215, 0, 255],
  [255, 77, 255, 255], [255, 140, 0, 255], [77, 124, 254, 255], [0, 230, 118, 255],
  [255, 111, 181, 255], [179, 136, 255, 255],
];

const tensor = (dimensions, data_type = 6, usage = 6, extra = {}) => ({
  dimensions, channels: 1, data_type, is_placeholder: false, usage, ...extra,
});
const ref = tensorName => ({ tensor: tensorName });
const named = (name, tensorName) => ({ name, tensor: tensorName });

const tensors = {
  vst_left: tensor([640, 640], 1, 6, { channels: 3 }),
  vst_right: tensor([640, 640], 1, 6, { channels: 3 }),
  vst_timestamp: tensor([1], 5, 5, { channels: 4 }),
  vst_camera_matrix: tensor([3, 3]),
  image_f32: tensor([640, 640], 6, 6, { channels: 3 }),
  model_output: tensor([8400, 85]),
  candidate_scores: tensor([CANDIDATES, 1]),
  candidate_boxes: tensor([CANDIDATES, 4]),
  candidate_classes: tensor([CANDIDATES, 1], 5),
  boxes_for_nms: tensor([CANDIDATES, 4]),
  offset_nms_boxes: tensor([COUNT, 4]),
  nms_scores: tensor([COUNT, 1]),
  nms_indices: tensor([COUNT, 1], 5),
  confirmation_state: tensor([COUNT, 6], 6, 6, { data: new Array(COUNT * 6).fill(0) }),
  next_state: tensor([COUNT, 6]),
  stable_boxes: tensor([COUNT, 4]),
  stable_scores: tensor([COUNT, 1]),
  stable_classes: tensor([COUNT, 1], 5),
  image_points_mat: tensor([COUNT, 2]),
  image_points: tensor([COUNT], 5, 1, { channels: 2 }),
  point_xyz_array: tensor([COUNT], 6, 1, { channels: 3 }),
  point_xyz: tensor([COUNT, 3]),
  right_eye_transform: tensor([4, 4]),
  left_eye_transform: tensor([4, 4]),
  detection: tensor([COUNT, 6], 6, 6, { is_placeholder: true }),
  node_poses: tensor([COUNT * 2, 4, 4], 6, 6, { is_placeholder: true }),
};

const classAwareScript =
  `var candidate_boxes;var candidate_classes;var boxes_for_nms;` +
  `for(var i=0;i<${CANDIDATES};i++){var offset=candidate_classes[i]*1000.0;` +
  `for(var c=0;c<4;c++)boxes_for_nms[i*4+c]=candidate_boxes[i*4+c]+offset;}`;

const stabilizationScript =
  `var candidate_boxes;var candidate_classes;var nms_scores;var nms_indices;var state;` +
  `var next_state;var stable_boxes;var stable_scores;var stable_classes;var used=[];` +
  `for(var p=0;p<${COUNT};p++)used[p]=false;` +
  `for(var z=0;z<${COUNT};z++){for(var c=0;c<6;c++)next_state[z*6+c]=0;` +
  `for(var b=0;b<4;b++)stable_boxes[z*4+b]=0;stable_scores[z]=0;stable_classes[z]=-1;}` +
  `for(var i=0;i<${COUNT};i++){var source=nms_indices[i];var score=nms_scores[i];` +
  `if(source<0||source>=${CANDIDATES}||score<=${THRESHOLD})continue;` +
  `var class_id=candidate_classes[source];var o=source*4;` +
  `var ax1=candidate_boxes[o];var ay1=candidate_boxes[o+1];` +
  `var ax2=candidate_boxes[o+2];var ay2=candidate_boxes[o+3];` +
  `if(!isFinite(ax1)||!isFinite(ay1)||!isFinite(ax2)||!isFinite(ay2))continue;` +
  `ax1=Math.max(0,Math.min(639,ax1));ay1=Math.max(0,Math.min(639,ay1));` +
  `ax2=Math.max(0,Math.min(639,ax2));ay2=Math.max(0,Math.min(639,ay2));` +
  `if(ax2<=ax1||ay2<=ay1)continue;` +
  `var best=-1;var best_iou=${CONFIRMATION_IOU};` +
  `for(var q=0;q<${COUNT};q++){if(used[q]||state[q*6+5]<=0||state[q*6+4]!==class_id)continue;` +
  `var bx1=state[q*6];var by1=state[q*6+1];var bx2=state[q*6+2];var by2=state[q*6+3];` +
  `var ix1=Math.max(ax1,bx1);var iy1=Math.max(ay1,by1);` +
  `var ix2=Math.min(ax2,bx2);var iy2=Math.min(ay2,by2);` +
  `var iw=Math.max(0,ix2-ix1);var ih=Math.max(0,iy2-iy1);var intersection=iw*ih;` +
  `var area_a=Math.max(0,ax2-ax1)*Math.max(0,ay2-ay1);` +
  `var area_b=Math.max(0,bx2-bx1)*Math.max(0,by2-by1);` +
  `var union_area=area_a+area_b-intersection;var overlap=union_area>0?intersection/union_area:0;` +
  `if(overlap>best_iou){best_iou=overlap;best=q;}}` +
  `var hits=1;if(best>=0){used[best]=true;hits=Math.min(${CONFIRMATION_FRAMES},state[best*6+5]+1);}` +
  `var clipped=[ax1,ay1,ax2,ay2];` +
  `for(var d=0;d<4;d++){var value=clipped[d];next_state[i*6+d]=value;stable_boxes[i*4+d]=value;}` +
  `next_state[i*6+4]=class_id;next_state[i*6+5]=hits;stable_classes[i]=class_id;` +
  `stable_scores[i]=hits>=${CONFIRMATION_FRAMES}?score:0;}`;

const pointScript =
  `var stable_boxes;var image_points_mat;for(var i=0;i<${COUNT};i++){var b=i*4;var p=i*2;` +
  `image_points_mat[p]=(stable_boxes[b]+stable_boxes[b+2])*0.5;` +
  `image_points_mat[p+1]=(stable_boxes[b+1]+stable_boxes[b+3])*0.5;}`;

const poseScript = `
var stable_boxes;var stable_scores;var stable_classes;var point_xyz;var vst_camera_matrix;
var left_eye_transform;var detection;var node_poses;
var placed_min_x=[],placed_max_x=[],placed_min_y=[],placed_max_y=[];
for(var i=0;i<${COUNT};i++){
  var b=i*4,p=i*3,d=i*6;
  var xmin=stable_boxes[b],ymin=stable_boxes[b+1],xmax=stable_boxes[b+2],ymax=stable_boxes[b+3];
  var width=xmax-xmin,height=ymax-ymin,score=stable_scores[i],class_id=stable_classes[i];
  detection[d]=(xmin+xmax)*0.5;detection[d+1]=(ymin+ymax)*0.5;
  detection[d+2]=width;detection[d+3]=height;detection[d+4]=score;detection[d+5]=class_id;
  var x=point_xyz[p],y=point_xyz[p+1],z=point_xyz[p+2];
  var has_depth=isFinite(x)&&isFinite(y)&&isFinite(z)&&z<-0.001;
  var render_depth=has_depth?Math.max(-5.0,Math.min(z,-1.5)):-1.5;
  var ratio=has_depth?render_depth/z:0.0;
  if(has_depth){x*=ratio;y*=ratio;}else{x=0.0;y=0.0;}z=render_depth;
  var positive_depth=-render_depth,fx=vst_camera_matrix[0],fy=vst_camera_matrix[4];
  var visible=score>${THRESHOLD}&&has_depth?1.0:0.0;
  var frame_scale_x=positive_depth*width/(fx*10.0)*visible;
  var frame_scale_y=positive_depth*height/(fy*10.0)*visible;
  var frame_local=[frame_scale_x,0,0,x+0.1,0,frame_scale_y,0,-y,0,0,0.05*visible,z,0,0,0,1];
  var frame_world=new Array(16).fill(0);
  for(var r=0;r<4;r++)for(var c=0;c<4;c++)for(var k=0;k<4;k++)
    frame_world[r*4+c]+=left_eye_transform[r*4+k]*frame_local[k*4+c];
  frame_world[1]=0;frame_world[4]=0;frame_world[6]=0;frame_world[9]=0;
  for(var m=0;m<16;m++)node_poses[i*16+m]=frame_world[m];

  var label_width=Math.min(112,Math.max(80,width)),label_height=label_width*0.25;
  var horizontal_sign=i%2===0?1.0:-1.0;
  var delta_u=(label_width-width)*0.5*horizontal_sign;
  var center_x=(xmin+xmax)*0.5+delta_u;
  var top_min_x=center_x-label_width*0.5,top_max_x=center_x+label_width*0.5;
  var top_min_y=ymin,top_max_y=ymin+label_height,collision=false;
  for(var previous=0;previous<i;previous++){
    if(stable_scores[previous]<=${THRESHOLD})continue;
    if(top_min_x<placed_max_x[previous]&&top_max_x>placed_min_x[previous]&&
       top_min_y<placed_max_y[previous]&&top_max_y>placed_min_y[previous])collision=true;
  }
  placed_min_x[i]=top_min_x;placed_max_x[i]=top_max_x;
  placed_min_y[i]=top_min_y;placed_max_y[i]=top_max_y;
  var vertical_sign=collision?-1.0:1.0;
  var delta_v=(label_height-height)*0.5*vertical_sign;
  var delta_x=positive_depth*delta_u/fx,delta_y=positive_depth*delta_v/fy;
  var label_scale_x=positive_depth*label_width/fx*visible;
  var label_scale_y=positive_depth*label_height/fy*visible;
  var label_local=[label_scale_x,0,0,x+delta_x+0.1,0,label_scale_y,0,-(y+delta_y),
                   0,0,0.01*visible,z+0.02,0,0,0,1];
  var label_world=new Array(16).fill(0);
  for(var rr=0;rr<4;rr++)for(var cc=0;cc<4;cc++)for(var kk=0;kk<4;kk++)
    label_world[rr*4+cc]+=left_eye_transform[rr*4+kk]*label_local[kk*4+cc];
  label_world[1]=0;label_world[4]=0;label_world[6]=0;label_world[9]=0;
  for(var n=0;n<16;n++)node_poses[(${COUNT}+i)*16+n]=label_world[n];
}`.replace(/\s+/g, ' ');

const inference = {
  tensors,
  operators: [
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO', inputs: [],
      outputs: [ref('vst_left'), ref('vst_right'), ref('vst_timestamp'), ref('vst_camera_matrix')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO', inputs: [ref('vst_left')], outputs: [ref('image_f32')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO', inputs: [named('images', 'image_f32')],
      outputs: [named('output0', 'model_output')], model: {
        bin_path: 'model/yolox_m.tflite', model_name: 'yolox_m', model_type: 'tflite', model_target: 'npu',
        input: [{ name: 'images', shape: [1, 640, 640, 3], encoding_type: 'FP32' }],
        output: [{ name: 'output0', shape: [1, 8400, 85], encoding_type: 'FP32' }],
      } },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO',
      inputs: [named('predictions', 'model_output')],
      outputs: [named('candidate_scores', 'candidate_scores'), named('candidate_boxes', 'candidate_boxes'),
                named('candidate_classes', 'candidate_classes')], model: {
        bin_path: 'model/yolox_postprocess.tflite', model_name: 'yolox_postprocess', model_type: 'tflite',
        model_target: 'cpu', cpu_target_num_threads: 4,
        input: [{ name: 'predictions', shape: [1, 8400, 85], encoding_type: 'FP32' }],
        output: [
          { name: 'candidate_scores', shape: [CANDIDATES, 1], encoding_type: 'FP32' },
          { name: 'candidate_boxes', shape: [CANDIDATES, 4], encoding_type: 'FP32' },
          { name: 'candidate_classes', shape: [CANDIDATES, 1], encoding_type: 'INT32' },
        ],
      } },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO',
      inputs: [named('candidate_boxes', 'candidate_boxes'), named('candidate_classes', 'candidate_classes')],
      outputs: [named('boxes_for_nms', 'boxes_for_nms')], attrs: [classAwareScript] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_NMS_PICO', attrs: ['0.5'],
      inputs: [ref('candidate_scores'), ref('boxes_for_nms')],
      outputs: [ref('nms_scores'), ref('offset_nms_boxes'), ref('nms_indices')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO',
      inputs: [named('candidate_boxes', 'candidate_boxes'), named('candidate_classes', 'candidate_classes'),
               named('nms_scores', 'nms_scores'), named('nms_indices', 'nms_indices'), named('state', 'confirmation_state')],
      outputs: [named('next_state', 'next_state'), named('stable_boxes', 'stable_boxes'),
                named('stable_scores', 'stable_scores'), named('stable_classes', 'stable_classes')],
      attrs: [stabilizationScript] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO', inputs: [ref('next_state')], outputs: [ref('confirmation_state')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO', inputs: [named('stable_boxes', 'stable_boxes')],
      outputs: [named('image_points_mat', 'image_points_mat')], attrs: [pointScript] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO', inputs: [ref('image_points_mat')], outputs: [ref('image_points')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO',
      inputs: [ref('image_points'), ref('vst_timestamp'), ref('vst_camera_matrix'), ref('vst_left'), ref('vst_right')],
      outputs: [ref('point_xyz_array')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO', inputs: [ref('point_xyz_array')], outputs: [ref('point_xyz')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO', inputs: [ref('vst_timestamp')],
      outputs: [ref('right_eye_transform'), ref('left_eye_transform')] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO',
      inputs: [named('stable_boxes', 'stable_boxes'), named('stable_scores', 'stable_scores'),
               named('stable_classes', 'stable_classes'), named('point_xyz', 'point_xyz'),
               named('vst_camera_matrix', 'vst_camera_matrix'), named('left_eye_transform', 'left_eye_transform')],
      outputs: [named('detection', 'detection'), named('node_poses', 'node_poses')], attrs: [poseScript] },
  ],
  inputs: [],
  outputs: ['detection', 'node_poses'],
};

const detectionTensor = tensor([COUNT, 6], 6, 6, { is_placeholder: true });
const gltfTensor = tensor([1, 1], 1, 7, {
  is_placeholder: true, is_gltf: true, asset: 'gltf/frame2_multi_10_pipeline.gltf',
});
const labelsTensors = {
  detection: detectionTensor,
  detection_gltf: gltfTensor,
  text_start: tensor([1], 6, 1, { channels: 2, data: [0.03, 0.65] }),
  text_font: tensor([1], 6, 2, { data: [46] }),
};
for (let i = 0; i < COUNT; i++) {
  labelsTensors[`class_text${i}`] = tensor([LABEL_LENGTH], 2, 2);
  labelsTensors[`text_colors${i}`] = tensor([2], 1, 4, {
    channels: 4, data: [...COLORS[i], 24, 24, 24, 224],
  });
  labelsTensors[`texture_id${i}`] = tensor([1], 3, 2, { data: [i] });
}

let labelScript = `var detection;${Array.from({ length: COUNT }, (_, i) => `var class_text${i};`).join('')}` +
  `var labels=${JSON.stringify(labels)};`;
for (let i = 0; i < COUNT; i++) {
  labelScript += `var o=${i * 6};var class_id=detection[o+5];` +
    `var valid=class_id>=0&&class_id<labels.length&&class_id===Math.floor(class_id);` +
    `var confidence=Math.max(0,Math.min(100,Math.round(detection[o+4]*100.0)));` +
    `var text=valid&&detection[o+4]>${THRESHOLD}?labels[class_id]+" "+confidence+"%":"";` +
    `for(var j=0;j<${LABEL_LENGTH};j++)class_text${i}[j]=j<text.length?text.charCodeAt(j):32;`;
}

const labelOutputs = Array.from({ length: COUNT }, (_, i) => named(`class_text${i}`, `class_text${i}`));
const labelsPipeline = {
  tensors: labelsTensors,
  operators: [
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_JS_SCRIPTING_PICO', inputs: [named('detection', 'detection')],
      outputs: labelOutputs, attrs: [labelScript] },
    ...Array.from({ length: COUNT }, (_, i) => ({
      type: 'XR_SECURE_MR_OPERATOR_TYPE_RENDER_TEXT_PICO',
      inputs: [ref(`class_text${i}`), ref('text_start'), ref(`text_colors${i}`), ref('detection_gltf'),
               ref(`texture_id${i}`), ref('text_font')],
      outputs: [], attrs: ['bold#en-US#512#128'],
    })),
  ],
  inputs: ['detection'],
  outputs: [],
};

const display = {
  tensors: {
    node_poses: tensor([COUNT * 2, 4, 4], 6, 6, { is_placeholder: true }),
    detection_gltf: gltfTensor,
    node_ids: tensor([COUNT * 2], 3, 2, { data: Array.from({ length: COUNT * 2 }, (_, i) => i) }),
    identity: tensor([4, 4], 6, 6, { data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }),
    visible: tensor([1], 2, 2, { data: [1] }),
  },
  operators: [
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_UPDATE_GLTF_PICO',
      inputs: [ref('detection_gltf'), ref('node_ids'), ref('node_poses')], outputs: [], attrs: ['local'] },
    { type: 'XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO',
      inputs: [ref('detection_gltf'), ref('identity'), ref('visible'), null], outputs: [] },
  ],
  inputs: ['node_poses'],
  outputs: [],
};

const manifest = {
  schema_version: '2',
  id: 'yolo_det_pipeline',
  pipelines: [
    { id: 'inference', path: 'pipeline/yolo_det_inference_pipeline.json' },
    { id: 'labels', path: 'pipeline/yolo_det_label_pipeline.json' },
    { id: 'display', path: 'pipeline/yolo_det_display_pipeline.json' },
  ],
  runtime: { supported_modes: ['xr'], detection_tensor: 'detection' },
};

mkdirSync(pipelineDir, { recursive: true });
mkdirSync(dirname(pipelineGltfPath), { recursive: true });
const pipelineGltf = JSON.parse(readFileSync(programmaticGltfPath, 'utf8'));
for (const node of pipelineGltf.nodes) {
  if (Number.isInteger(node.mesh)) {
    // A node must remain invisible until UPDATE_GLTF supplies its first valid
    // local pose. Otherwise an uninitialized frame/label renders at the world
    // origin using the template's large authored geometry.
    node.scale = [0, 0, 0];
  }
}
writeFileSync(pipelineGltfPath, JSON.stringify(pipelineGltf, null, 2) + '\n');
writeFileSync(resolve(pipelineDir, 'yolo_det_inference_pipeline.json'), JSON.stringify(inference, null, 2) + '\n');
writeFileSync(resolve(pipelineDir, 'yolo_det_label_pipeline.json'), JSON.stringify(labelsPipeline, null, 2) + '\n');
writeFileSync(resolve(pipelineDir, 'yolo_det_display_pipeline.json'), JSON.stringify(display, null, 2) + '\n');
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

console.log(`Generated YOLO package JSON for ${COUNT} detections.`);
