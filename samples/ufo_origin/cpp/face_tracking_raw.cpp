#include "face_tracking_raw.h"
#include "helper.h"

namespace SecureMR {

FaceTrackingRaw::FaceTrackingRaw(const XrInstance &instance, const XrSession &session)
    : xr_instance(instance), xr_session(session) {
  getInstanceProcAddr();
}

FaceTrackingRaw::~FaceTrackingRaw() {
  keepRunning = false;
  if (pipelineInitializer && pipelineInitializer->joinable()) {
    pipelineInitializer->join();
  }
  for (auto &runner : pipelineRunners) {
    if (runner.joinable()) runner.join();
  }

  if (m_secureMrVSTImagePipeline != XR_NULL_HANDLE) {
    xrDestroySecureMrPipelinePICO(m_secureMrVSTImagePipeline);
  }
  if (m_secureMrModelInferencePipeline != XR_NULL_HANDLE) {
    xrDestroySecureMrPipelinePICO(m_secureMrModelInferencePipeline);
  }
  if (m_secureMrMap2dTo3dPipeline != XR_NULL_HANDLE) {
    xrDestroySecureMrPipelinePICO(m_secureMrMap2dTo3dPipeline);
  }
  if (m_secureMrRenderingPipeline != XR_NULL_HANDLE) {
    xrDestroySecureMrPipelinePICO(m_secureMrRenderingPipeline);
  }
  if (m_secureMrFramework != XR_NULL_HANDLE) {
    xrDestroySecureMrFrameworkPICO(m_secureMrFramework);
  }
}

void FaceTrackingRaw::CreateFramework() {
  Log::Write(Log::Level::Info, "CreateFramework ...");
  XrSecureMrFrameworkCreateInfoPICO createInfo{XR_TYPE_SECURE_MR_FRAMEWORK_CREATE_INFO_PICO, nullptr, 256, 256};
  CHECK_XRCMD(xrCreateSecureMrFrameworkPICO(xr_session, &createInfo, &m_secureMrFramework));
  Log::Write(Log::Level::Info, "CreateFramework done.");
}

void FaceTrackingRaw::CreatePipelines() {
  pipelineInitializer = std::make_unique<std::thread>([this]() {
    CreateSecureMrVSTImagePipeline();
    CreateSecureMrModelInferencePipeline();
    CreateSecureMrMap2dTo3dPipeline();
    CreateSecureMrRenderingPipeline();

    initialized.notify_all();
    pipelineAllInitialized = true;
  });
}

void FaceTrackingRaw::RunPipelines() {
  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrVSTImagePipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrModelInferencePipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrMap2dTo3dPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  pipelineRunners.emplace_back([this]() {
    {
      std::unique_lock<std::mutex> guard(initialized_mtx);
      initialized.wait(guard);
    }
    while (keepRunning) {
      RunSecureMrRenderingPipeline();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });
}

void FaceTrackingRaw::GetVSTImages() {
  Log::Write(Log::Level::Info, "Open MR: GetVSTImages");

  // get VST access
  XrSecureMrOperatorPICO vstOperator;
  m_apiHelper->CreateOperator(m_secureMrVSTImagePipeline, vstOperator,
                              XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO);
  m_apiHelper->SetOutput(m_secureMrVSTImagePipeline, vstOperator, vstOutputLeftUint8Placeholder, "left image");
  m_apiHelper->SetOutput(m_secureMrVSTImagePipeline, vstOperator, vstOutputRightUint8Placeholder, "right image");
  m_apiHelper->SetOutput(m_secureMrVSTImagePipeline, vstOperator, vstTimestampPlaceholder, "timestamp");
  m_apiHelper->SetOutput(m_secureMrVSTImagePipeline, vstOperator, vstCameraMatrixPlaceholder, "camera matrix");

  // use assignment operator to convert vstOutputTensor to float tensor
  XrSecureMrOperatorPICO assignmentOperator;
  m_apiHelper->CreateOperator(m_secureMrVSTImagePipeline, assignmentOperator,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrVSTImagePipeline, assignmentOperator, vstOutputLeftUint8Placeholder, "src");
  m_apiHelper->SetOutput(m_secureMrVSTImagePipeline, assignmentOperator, vstOutputLeftFp32Placeholder, "dst");

  // use arithmetic operator to divide by 255.0f, config is {0} / 255.0f
  XrSecureMrOperatorPICO arithmeticOperator;
  m_apiHelper->CreateArithmeticOperator(m_secureMrVSTImagePipeline, arithmeticOperator, "{0} / 255.0");
  m_apiHelper->SetInput(m_secureMrVSTImagePipeline, arithmeticOperator, vstOutputLeftFp32Placeholder, "{0}");
  m_apiHelper->SetOutput(m_secureMrVSTImagePipeline, arithmeticOperator, vstOutputLeftFp32Placeholder, "result");
}

void FaceTrackingRaw::RunModelInference() {
  Log::Write(Log::Level::Info, "Open MR: RunModelInference");

  // Step1: run model inference - input: vstOutputTensorFp32, output: faceAnchor, faceScores
  XrSecureMrPipelineTensorPICO faceAnchor;
  int32_t faceAnchorDimensions[2] = {896, 16};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, faceAnchor, faceAnchorDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);
  XrSecureMrPipelineTensorPICO faceScores;
  int32_t dimensionsFaceScores[2] = {896, 1};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, faceScores, dimensionsFaceScores, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  // XrSecureMrPipelineTensorPICO vstImagePlaceholder;
  int32_t vstImageDimensions[2] = {256, 256};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, vstImagePlaceholder, vstImageDimensions, 2, 3,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    true);

  // Model info
  XrSecureMrOperatorIOMapPICO inputNode;
  inputNode.type = XR_TYPE_SECURE_MR_OPERATOR_IO_MAP_PICO;
  inputNode.next = nullptr;
  inputNode.encodingType = XR_SECURE_MR_MODEL_ENCODING_FLOAT_32_PICO;
  strncpy(inputNode.nodeName, "image", XR_MAX_OPERATOR_NODE_NAME_PICO);
  strncpy(inputNode.operatorIOName, "input_rgb", XR_MAX_OPERATOR_NODE_NAME_PICO);
  XrSecureMrOperatorIOMapPICO outputNodeFaceAnchor;
  outputNodeFaceAnchor.type = XR_TYPE_SECURE_MR_OPERATOR_IO_MAP_PICO;
  outputNodeFaceAnchor.next = nullptr;
  outputNodeFaceAnchor.encodingType = XR_SECURE_MR_MODEL_ENCODING_FLOAT_32_PICO;
  strncpy(outputNodeFaceAnchor.nodeName, "box_coords", XR_MAX_OPERATOR_NODE_NAME_PICO);
  strncpy(outputNodeFaceAnchor.operatorIOName, "face_anchor", XR_MAX_OPERATOR_NODE_NAME_PICO);
  XrSecureMrOperatorIOMapPICO outputNodeScore;
  outputNodeScore.type = XR_TYPE_SECURE_MR_OPERATOR_IO_MAP_PICO;
  outputNodeScore.next = nullptr;
  outputNodeScore.encodingType = XR_SECURE_MR_MODEL_ENCODING_FLOAT_32_PICO;
  strncpy(outputNodeScore.nodeName, "box_scores", XR_MAX_OPERATOR_NODE_NAME_PICO);
  strncpy(outputNodeScore.operatorIOName, "score", XR_MAX_OPERATOR_NODE_NAME_PICO);

  std::vector<char> modelData;
  XrSecureMrOperatorModelPICO modelOperatorPico;
  if (m_apiHelper->LoadModelData(FACE_DETECTION_MODEL_PATH, modelData)) {
    modelOperatorPico.type = XR_TYPE_SECURE_MR_OPERATOR_MODEL_PICO;
    modelOperatorPico.next = nullptr;
    modelOperatorPico.modelInputs = &inputNode;
    modelOperatorPico.modelInputCount = 1;
    modelOperatorPico.modelOutputs = new XrSecureMrOperatorIOMapPICO[2]{outputNodeFaceAnchor, outputNodeScore};
    modelOperatorPico.modelOutputCount = 2;
    modelOperatorPico.bufferSize = static_cast<int32_t>(modelData.size());
    modelOperatorPico.buffer = modelData.data();
    modelOperatorPico.modelType = XR_SECURE_MR_MODEL_TYPE_QNN_CONTEXT_BINARY_PICO;
    modelOperatorPico.modelName = "face";
  } else {
    Log::Write(Log::Level::Error, "Failed to load model data from file.");
  }
  XrSecureMrOperatorPICO modelInferenceOperator;
  XrSecureMrOperatorCreateInfoPICO modelInferenceOperatorCreateInfo;
  modelInferenceOperatorCreateInfo.type = XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO;
  modelInferenceOperatorCreateInfo.operatorInfo =
      reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO *>(&modelOperatorPico);
  modelInferenceOperatorCreateInfo.operatorType = XR_SECURE_MR_OPERATOR_TYPE_RUN_MODEL_INFERENCE_PICO;
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_secureMrModelInferencePipeline, &modelInferenceOperatorCreateInfo,
                                           &modelInferenceOperator));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, modelInferenceOperator,
                                                     vstImagePlaceholder, "input_rgb"));
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_secureMrModelInferencePipeline, modelInferenceOperator,
                                                    faceAnchor, "face_anchor"));
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_secureMrModelInferencePipeline, modelInferenceOperator,
                                                    faceScores, "score"));

  // Step2: get face landmarks from face anchor - input: faceAnchor[896, 16], output: faceLandmarks[896, 12]
  XrSecureMrPipelineTensorPICO faceLandmarks;
  int32_t faceLandmarksDimensions[2] = {896, 4};  // left eye, right eye
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, faceLandmarks, faceLandmarksDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  // create pipeline tensor vector - slicePos
  XrSecureMrPipelineTensorPICO srcSliceTensor_1;
  srcSliceTensor_1 = m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0, 4}, {-1, 8}, {}, 2,
                                                              2 * 2 * sizeof(int32_t));
  XrSecureMrOperatorPICO anchorToSliceOperator_1;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, anchorToSliceOperator_1,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, anchorToSliceOperator_1, faceAnchor, "src");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, anchorToSliceOperator_1, srcSliceTensor_1, "src slices");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, anchorToSliceOperator_1, faceLandmarks, "dst");

  // Step3: apply anchor.mat to face landmarks
  XrSecureMrPipelineTensorPICO anchorMatTensor;
  int32_t anchorMatDimensions[2] = {896, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, anchorMatTensor, anchorMatDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  std::vector<char> anchorData;
  if (m_apiHelper->LoadModelData(ANCHOR_MAT, anchorData)) {
    XrSecureMrTensorBufferPICO anchorTensorBuffer = {};
    anchorTensorBuffer.type = XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO;
    anchorTensorBuffer.next = nullptr;
    anchorTensorBuffer.bufferSize = anchorData.size();
    anchorTensorBuffer.buffer = anchorData.data();
    CHECK_XRCMD(
        xrResetSecureMrPipelineTensorPICO(m_secureMrModelInferencePipeline, anchorMatTensor, &anchorTensorBuffer));
  } else {
    Log::Write(Log::Level::Error, "Failed to load anchor.mat data from file.");
  }
  // get fist two cols of anchor mat
  XrSecureMrPipelineTensorPICO anchorMatFirstTwoCols;
  int32_t anchorMatFirstTwoColsDimensions[2] = {896, 2};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, anchorMatFirstTwoCols,
                                    anchorMatFirstTwoColsDimensions, 2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);

  XrSecureMrPipelineTensorPICO srcSliceFirstTwoColsTensor;
  srcSliceFirstTwoColsTensor = m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0, 0},
                                                                        {-1, 2}, {}, 2, 2 * 2 * sizeof(int32_t));

  XrSecureMrOperatorPICO sliceOperatorAnchorMatFirstTwoCols;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, sliceOperatorAnchorMatFirstTwoCols,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, sliceOperatorAnchorMatFirstTwoCols, anchorMatTensor, "src");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, sliceOperatorAnchorMatFirstTwoCols,
                        srcSliceFirstTwoColsTensor, "src slices");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, sliceOperatorAnchorMatFirstTwoCols, anchorMatFirstTwoCols,
                         "dst");

  // duplicate anchor mat to [896, 12]
  XrSecureMrPipelineTensorPICO anchorMatDuplicated;
  int32_t anchorMatDuplicatedDimensions[2] = {896, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, anchorMatDuplicated,
                                    anchorMatDuplicatedDimensions, 2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);
  // copy first two cols
  XrSecureMrOperatorPICO assignmentOpAnchorMatDuplicated;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated, anchorMatFirstTwoCols,
                        "src");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated, srcSliceFirstTwoColsTensor,
                        "dst slices");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated, anchorMatDuplicated, "dst");

  // copy last two cols
  XrSecureMrOperatorPICO assignmentOpAnchorMatDuplicated_2;
  XrSecureMrPipelineTensorPICO srcSliceLastTwoColsTensor;
  srcSliceLastTwoColsTensor = m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0, 2},
                                                                       {-1, 4}, {}, 2, 2 * 2 * sizeof(int32_t));

  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated_2,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated_2, anchorMatTensor, "src");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated_2, srcSliceLastTwoColsTensor,
                        "dst slices");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, assignmentOpAnchorMatDuplicated_2, anchorMatDuplicated,
                         "dst");

  // faceLandmarksEyes = faceLandmarksEyes / 256 +  anchorMatDuplicated
  XrSecureMrOperatorPICO arithmeticOperatorFaceLandmarks;
  m_apiHelper->CreateArithmeticOperator(m_secureMrModelInferencePipeline, arithmeticOperatorFaceLandmarks,
                                        "({0} / 256.0 + {1}) * 256.0");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, arithmeticOperatorFaceLandmarks, faceLandmarks, "{0}");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, arithmeticOperatorFaceLandmarks, anchorMatDuplicated, "{1}");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, arithmeticOperatorFaceLandmarks, faceLandmarks, "result");

  // use argmax to get best face, input: faceScores[896, 1], output: bestFaceIndex[1]
  XrSecureMrPipelineTensorPICO bestFaceIndex;
  int32_t bestFaceIndexDimensions[1] = {1};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, bestFaceIndex, bestFaceIndexDimensions, 1, 2,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_SLICE_PICO,
                                    false);

  XrSecureMrOperatorPICO argmaxOperator;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, argmaxOperator, XR_SECURE_MR_OPERATOR_TYPE_ARGMAX_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, argmaxOperator, faceScores, "operand");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, argmaxOperator, bestFaceIndex, "result");

  XrSecureMrPipelineTensorPICO bestFaceIndexMat;
  int32_t bestFaceIndexMatDimensions[2] = {1, 1};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, bestFaceIndexMat, bestFaceIndexMatDimensions, 2,
                                    1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  XrSecureMrOperatorPICO assignmentOpBestFaceIndex;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, assignmentOpBestFaceIndex,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpBestFaceIndex, bestFaceIndex, "src");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, assignmentOpBestFaceIndex, bestFaceIndexMat, "dst");

  // index + 1
  XrSecureMrPipelineTensorPICO bestFaceIndexPlusOne;
  int32_t bestFaceIndexPlusOneDimensions[2] = {1, 1};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, bestFaceIndexPlusOne,
                                    bestFaceIndexPlusOneDimensions, 2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);

  XrSecureMrOperatorPICO arithmeticOperator_1;
  XrSecureMrOperatorArithmeticComposePICO arithmeticOperatorPico_1{XR_TYPE_SECURE_MR_OPERATOR_ARITHMETIC_COMPOSE_PICO,
                                                                   nullptr, "{0} + 1"};
  XrSecureMrOperatorCreateInfoPICO arithmeticOperatorCreateInfoPico_1{
      XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
      reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO *>(&arithmeticOperatorPico_1),
      XR_SECURE_MR_OPERATOR_TYPE_ARITHMETIC_COMPOSE_PICO};
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_secureMrModelInferencePipeline, &arithmeticOperatorCreateInfoPico_1,
                                           &arithmeticOperator_1));
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, arithmeticOperator_1, bestFaceIndexMat, "{0}");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, arithmeticOperator_1, bestFaceIndexPlusOne, "result");

  // copy best face landmark
  XrSecureMrPipelineTensorPICO srcSlicesBestFace;
  srcSlicesBestFace = m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0, 0}, {-1, 4}, {}, 2,
                                                               2 * 2 * sizeof(int32_t));
  XrSecureMrOperatorPICO assignmentOp_1;
  XrSecureMrOperatorBaseHeaderPICO assignmentOpInfo{XR_TYPE_SECURE_MR_OPERATOR_BASE_HEADER_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO assignmentOpCreateInfo{XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
                                                          &assignmentOpInfo,
                                                          XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO};
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_secureMrModelInferencePipeline, &assignmentOpCreateInfo, &assignmentOp_1));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOp_1, bestFaceIndexMat,
                                                     "src"));
  XrSecureMrPipelineTensorPICO dstSlicesBestFace;
  dstSlicesBestFace =
      m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0}, {1}, {}, 1, 2 * sizeof(int32_t));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOp_1,
                                                     dstSlicesBestFace, "dst channel slice"));
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_secureMrModelInferencePipeline, assignmentOp_1, srcSlicesBestFace,
                                                    "dst"));

  XrSecureMrOperatorPICO assignmentOp_2;
  XrSecureMrOperatorBaseHeaderPICO assignmentOpInfo_2{XR_TYPE_SECURE_MR_OPERATOR_BASE_HEADER_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO assignmentOpCreateInfo_2{XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
                                                            &assignmentOpInfo_2,
                                                            XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO};
  CHECK_XRCMD(
      xrCreateSecureMrOperatorPICO(m_secureMrModelInferencePipeline, &assignmentOpCreateInfo_2, &assignmentOp_2));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOp_2,
                                                     bestFaceIndexPlusOne, "src"));
  XrSecureMrPipelineTensorPICO dstSlicesBestFacePlusOne;
  dstSlicesBestFacePlusOne =
      m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {1}, {1}, {}, 1, 2 * sizeof(int32_t));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOp_2,
                                                     dstSlicesBestFacePlusOne, "dst channel slice"));
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_secureMrModelInferencePipeline, assignmentOp_2, srcSlicesBestFace,
                                                    "dst"));

  XrSecureMrPipelineTensorPICO bestFaceLandmark;
  int32_t bestFaceLandmarkDimensions[2] = {1, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, bestFaceLandmark, bestFaceLandmarkDimensions, 2,
                                    1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  XrSecureMrOperatorPICO assignmentOpBestFaceAnchors;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, assignmentOpBestFaceAnchors,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpBestFaceAnchors, faceLandmarks, "src");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpBestFaceAnchors, srcSlicesBestFace, "src slices");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, assignmentOpBestFaceAnchors, bestFaceLandmark, "dst");

  // bestFaceLandmarkInt32
  XrSecureMrPipelineTensorPICO bestFaceLandmarkInt32;
  int32_t bestFaceLandmarkInt32Dimensions[2] = {1, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, bestFaceLandmarkInt32,
                                    bestFaceLandmarkInt32Dimensions, 2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);

  XrSecureMrOperatorPICO assignmentOpBestFaceLandmarkInt32;
  m_apiHelper->CreateOperator(m_secureMrModelInferencePipeline, assignmentOpBestFaceLandmarkInt32,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, assignmentOpBestFaceLandmarkInt32, bestFaceLandmark, "src");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, assignmentOpBestFaceLandmarkInt32, bestFaceLandmarkInt32,
                         "dst");

  // leftEyeUV = bestFaceLandmark[0, 1]
  XrSecureMrOperatorPICO assignmentOpLeftEyeUV;
  XrSecureMrOperatorBaseHeaderPICO assignmentOpLeftEyeUVInfo{XR_TYPE_SECURE_MR_OPERATOR_BASE_HEADER_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO assignmentOpLeftEyeUVCreateInfo{XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
                                                                   &assignmentOpLeftEyeUVInfo,
                                                                   XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO};
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_secureMrModelInferencePipeline, &assignmentOpLeftEyeUVCreateInfo,
                                           &assignmentOpLeftEyeUV));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOpLeftEyeUV,
                                                     bestFaceLandmarkInt32, "src"));
  // src slice
  XrSecureMrPipelineTensorPICO srcSliceLeftEyeUV;
  srcSliceLeftEyeUV = m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0, 0}, {1, 2}, {}, 2,
                                                               2 * 2 * sizeof(int32_t));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOpLeftEyeUV,
                                                     srcSliceLeftEyeUV, "src slices"));
  XrSecureMrPipelineTensorPICO dstChannelSliceLeftEyeUV;
  dstChannelSliceLeftEyeUV =
      m_apiHelper->CreatePipelineTensorAsSlice(m_secureMrModelInferencePipeline, {0}, {2}, {}, 1, 2 * sizeof(int32_t));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrModelInferencePipeline, assignmentOpLeftEyeUV,
                                                     dstChannelSliceLeftEyeUV, "dst channel slice"));
  CHECK_XRCMD(xrSetSecureMrOperatorResultByNamePICO(m_secureMrModelInferencePipeline, assignmentOpLeftEyeUV,
                                                    leftEyeUVPlaceholder, "dst"));

  // compare leftEyeUVPlaceholder with threshold[10, 10]
  // threshold
  XrSecureMrPipelineTensorPICO thresholdLeftEyeUV;
  int32_t thresholdLeftEyeUVDimensions[1] = {1};
  int32_t thresholdLeftEyeUVData[2] = {10, 10};
  m_apiHelper->CreateAndSetPipelineTensor(m_secureMrModelInferencePipeline, thresholdLeftEyeUV,
                                          thresholdLeftEyeUVDimensions, 1, 2, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO,
                                          XR_SECURE_MR_TENSOR_TYPE_POINT_PICO, thresholdLeftEyeUVData,
                                          sizeof(thresholdLeftEyeUVData), false);

  XrSecureMrOperatorPICO compareOpLeftEyeUV;
  XrSecureMrOperatorComparisonPICO compareOpLeftEyeUVInfo{XR_TYPE_SECURE_MR_OPERATOR_COMPARISON_PICO, nullptr,
                                                          XR_SECURE_MR_COMPARISON_LARGER_THAN_PICO};
  XrSecureMrOperatorCreateInfoPICO compareOpLeftEyeUVCreateInfo{
      XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
      reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO *>(&compareOpLeftEyeUVInfo),
      XR_SECURE_MR_OPERATOR_TYPE_CUSTOMIZED_COMPARE_PICO};
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_secureMrModelInferencePipeline, &compareOpLeftEyeUVCreateInfo,
                                           &compareOpLeftEyeUV));
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, compareOpLeftEyeUV, leftEyeUVPlaceholder, "operand0");
  m_apiHelper->SetInput(m_secureMrModelInferencePipeline, compareOpLeftEyeUV, thresholdLeftEyeUV, "operand1");
  m_apiHelper->SetOutput(m_secureMrModelInferencePipeline, compareOpLeftEyeUV, isFaceDetectedPlaceholder, "result");
}

void FaceTrackingRaw::Map2Dto3D() {
  Log::Write(Log::Level::Info, "Open MR: CreateSecureMRPipeline");

  // Convert UV to 3D Points
  XrSecureMrPipelineTensorPICO leftEyeXYZ;
  int32_t points3DDimensions[1] = {1};
  float leftEyeXYZData[3] = {0.0, 0.0, 0.0};
  m_apiHelper->CreateAndSetPipelineTensor(
      m_secureMrMap2dTo3dPipeline, leftEyeXYZ, points3DDimensions, 1, 3, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO,
      XR_SECURE_MR_TENSOR_TYPE_POINT_PICO, leftEyeXYZData, sizeof(leftEyeXYZData), false);

  XrSecureMrOperatorPICO uvTo3DOperator;
  XrSecureMrOperatorUVTo3DPICO uvTo3DOperatorPico{XR_TYPE_SECURE_MR_OPERATOR_UV_TO_3D_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO uvTo3DCreateInfoPico{
      XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr,
      reinterpret_cast<XrSecureMrOperatorBaseHeaderPICO *>(&uvTo3DOperatorPico),
      XR_SECURE_MR_OPERATOR_TYPE_UV_TO_3D_IN_CAM_SPACE_PICO};
  CHECK_XRCMD(xrCreateSecureMrOperatorPICO(m_secureMrMap2dTo3dPipeline, &uvTo3DCreateInfoPico, &uvTo3DOperator));
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, uvTo3DOperator, uvPlaceholder, "uv");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, uvTo3DOperator, timestampPlaceholder, "timestamp");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, uvTo3DOperator, cameraMatrixPlaceholder, "camera intrisic");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, uvTo3DOperator, leftImgePlaceholder, "left image");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, uvTo3DOperator, rightImagePlaceholder, "right image");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, uvTo3DOperator, leftEyeXYZ, "point_xyz");

  // copy leftEyeXYZ to leftEyeXYZMat
  XrSecureMrPipelineTensorPICO leftEyeXYZMat;
  int32_t leftEyeXYZMatDimensions[2] = {3, 1};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, leftEyeXYZMat, leftEyeXYZMatDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  XrSecureMrOperatorPICO assignmentOpLeftEyeXYZ;
  m_apiHelper->CreateOperator(m_secureMrMap2dTo3dPipeline, assignmentOpLeftEyeXYZ,
                              XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, assignmentOpLeftEyeXYZ, leftEyeXYZ, "src");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, assignmentOpLeftEyeXYZ, leftEyeXYZMat, "dst");

  // leftEyeXYZMat = leftEyeXYZMat * [1, 1, -1] - use elememtwise multiplication
  // create [1, 1, -1] tensor
  XrSecureMrPipelineTensorPICO leftEyeXYZMatMultiplier;
  int32_t leftEyeXYZMatMultiplierDimensions[2] = {3, 1};
  float leftEyeXYZMatMultiplierData[3] = {1.0, -1.0, 1.0};
  m_apiHelper->CreateAndSetPipelineTensor(m_secureMrMap2dTo3dPipeline, leftEyeXYZMatMultiplier,
                                          leftEyeXYZMatMultiplierDimensions, 2, 1,
                                          XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                          leftEyeXYZMatMultiplierData, sizeof(leftEyeXYZMatMultiplierData), false);
  // elementwise multiplication
  XrSecureMrOperatorPICO elementwiseMultiplicationOperator;
  m_apiHelper->CreateOperator(m_secureMrMap2dTo3dPipeline, elementwiseMultiplicationOperator,
                              XR_SECURE_MR_OPERATOR_TYPE_ELEMENTWISE_MULTIPLY_PICO);
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, elementwiseMultiplicationOperator, leftEyeXYZMat, "operand0");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, elementwiseMultiplicationOperator, leftEyeXYZMatMultiplier,
                        "operand1");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, elementwiseMultiplicationOperator, leftEyeXYZMat, "result");

  // add offset to leftEyeXYZMatAdjusted = leftEyeXYZMatAdjusted + [0, 0.2, 0.0]
  // create [0, 0.2, 0.0] tensor
  XrSecureMrPipelineTensorPICO leftEyeXYZMatOffset;
  int32_t leftEyeXYZMatOffsetDimensions[2] = {3, 1};
  float leftEyeXYZMatOffsetData[3] = {0.05, 0.25, -0.05};
  m_apiHelper->CreateAndSetPipelineTensor(m_secureMrMap2dTo3dPipeline, leftEyeXYZMatOffset,
                                          leftEyeXYZMatOffsetDimensions, 2, 1,
                                          XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                          leftEyeXYZMatOffsetData, sizeof(leftEyeXYZMatOffsetData), false);
  XrSecureMrOperatorPICO arithmeticOperator;
  m_apiHelper->CreateArithmeticOperator(m_secureMrMap2dTo3dPipeline, arithmeticOperator, "{0} + {1}");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, arithmeticOperator, leftEyeXYZMat, "{0}");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, arithmeticOperator, leftEyeXYZMatOffset, "{1}");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, arithmeticOperator, leftEyeXYZMat, "result");

  // 2:  Convert 3D Points to World Coordinates
  XrSecureMrPipelineTensorPICO rvecTensor;
  int32_t rvecTensorDimensions[2] = {3, 1};
  float rvecData[3] = {0, 0, 0};
  m_apiHelper->CreateAndSetPipelineTensor(m_secureMrMap2dTo3dPipeline, rvecTensor, rvecTensorDimensions, 2, 1,
                                          XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                          rvecData, sizeof(rvecData), false);

  XrSecureMrPipelineTensorPICO svecTensor;
  int32_t svecTensorDimensions[2] = {3, 1};
  float svecData[3] = {0.1, 0.1, 0.1};
  m_apiHelper->CreateAndSetPipelineTensor(m_secureMrMap2dTo3dPipeline, svecTensor, svecTensorDimensions, 2, 1,
                                          XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                          svecData, sizeof(svecData), false);

  XrSecureMrPipelineTensorPICO pipelineResultTensor;
  int32_t pipelineResultTensorDimensions[2] = {4, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, pipelineResultTensor, pipelineResultTensorDimensions,
                                    2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  XrSecureMrOperatorPICO makeTransformMatOperator;
  m_apiHelper->CreateOperator(m_secureMrMap2dTo3dPipeline, makeTransformMatOperator,
                              XR_SECURE_MR_OPERATOR_TYPE_GET_TRANSFORM_MAT_PICO);
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, makeTransformMatOperator, rvecTensor, "rotation");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, makeTransformMatOperator, leftEyeXYZMat, "translation");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, makeTransformMatOperator, svecTensor, "scale");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, makeTransformMatOperator, pipelineResultTensor, "result");

  // get cam to world matrix
  XrSecureMrPipelineTensorPICO camToWorldMat;
  int32_t camToWorldMatDimensions[2] = {4, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, camToWorldMat, camToWorldMatDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);
  XrSecureMrOperatorPICO camtToWorldOp;
  m_apiHelper->CreateOperator(m_secureMrMap2dTo3dPipeline, camtToWorldOp,
                              XR_SECURE_MR_OPERATOR_TYPE_CAMERA_SPACE_TO_WORLD_PICO);
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, camtToWorldOp, timestampPlaceholder, "timestamp");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, camtToWorldOp, camToWorldMat, "left");

  XrSecureMrOperatorPICO mulOpCurr;
  m_apiHelper->CreateArithmeticOperator(m_secureMrMap2dTo3dPipeline, mulOpCurr, "{0} * {1}");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, mulOpCurr, camToWorldMat, "{0}");
  m_apiHelper->SetInput(m_secureMrMap2dTo3dPipeline, mulOpCurr, pipelineResultTensor, "{1}");
  m_apiHelper->SetOutput(m_secureMrMap2dTo3dPipeline, mulOpCurr, currentPositionPlaceholder, "result");
}

void FaceTrackingRaw::CreateSecureMrVSTImagePipeline() {
  Log::Write(Log::Level::Info, "SpatialML CreateSecureMrVSTImagePipeline");
  XrSecureMrPipelineCreateInfoPICO createInfo{XR_TYPE_SECURE_MR_PIPELINE_CREATE_INFO_PICO, nullptr};
  CHECK_XRCMD(xrCreateSecureMrPipelinePICO(m_secureMrFramework, &createInfo, &m_secureMrVSTImagePipeline));

  // global tensors for VST
  int32_t vstDimensions[2] = {256, 256};
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, vstOutputLeftUint8, vstDimensions, 2, 3,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, vstOutputRightUint8, vstDimensions, 2, 3,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, vstOutputLeftFp32, vstDimensions, 2, 3,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);
  // timestamp
  int32_t timestampDimensions[1] = {1};
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, vstTimestamp, timestampDimensions, 1, 4,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO,
                                  false);
  // camera matrix
  int32_t cameraMatrixDimensions[2] = {3, 3};
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, vstCameraMatrix, cameraMatrixDimensions, 2, 1,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);

  // placeholders
  // vst left, right - uint8
  m_apiHelper->CreatePipelineTensor(m_secureMrVSTImagePipeline, vstOutputLeftUint8Placeholder, vstDimensions, 2, 3,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, true);
  m_apiHelper->CreatePipelineTensor(m_secureMrVSTImagePipeline, vstOutputRightUint8Placeholder, vstDimensions, 2, 3,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, true);

  // vst left, right - float32
  m_apiHelper->CreatePipelineTensor(m_secureMrVSTImagePipeline, vstOutputLeftFp32Placeholder, vstDimensions, 2, 3,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    true);

  // timestamp
  m_apiHelper->CreatePipelineTensor(m_secureMrVSTImagePipeline, vstTimestampPlaceholder, timestampDimensions, 1, 4,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO,
                                    true);

  // camera matrix
  m_apiHelper->CreatePipelineTensor(m_secureMrVSTImagePipeline, vstCameraMatrixPlaceholder, cameraMatrixDimensions, 2,
                                    1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    true);

  GetVSTImages();
}

void FaceTrackingRaw::CreateSecureMrModelInferencePipeline() {
  Log::Write(Log::Level::Info, "SpatialML: CreateSecureMrModelInferencePipeline");
  XrSecureMrPipelineCreateInfoPICO createInfo{XR_TYPE_SECURE_MR_PIPELINE_CREATE_INFO_PICO, nullptr};
  CHECK_XRCMD(xrCreateSecureMrPipelinePICO(m_secureMrFramework, &createInfo, &m_secureMrModelInferencePipeline));

  int32_t leftEyeUVDimensions[1] = {1};
  int leftEyeUVGlobalData[2] = {0, 0};
  m_apiHelper->CreateAndSetGlobalTensor(m_secureMrFramework, leftEyeUVGlobal, leftEyeUVDimensions, 1, 2,
                                        XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                        leftEyeUVGlobalData, sizeof(leftEyeUVGlobalData));
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, leftEyeUVPlaceholder, leftEyeUVDimensions, 1, 2,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                    true);

  int32_t allResultLeftEyeUVDimensions[1] = {1};
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, isFaceDetected, allResultLeftEyeUVDimensions, 1, 2,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_POINT_PICO, false);
  m_apiHelper->CreatePipelineTensor(m_secureMrModelInferencePipeline, isFaceDetectedPlaceholder,
                                    allResultLeftEyeUVDimensions, 1, 2, XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_POINT_PICO, true);
  RunModelInference();
}

void FaceTrackingRaw::CreateSecureMrMap2dTo3dPipeline() {
  Log::Write(Log::Level::Info, "Open MR: CreateSecureMRPipeline");
  XrSecureMrPipelineCreateInfoPICO createInfo{XR_TYPE_SECURE_MR_PIPELINE_CREATE_INFO_PICO, nullptr};
  CHECK_XRCMD(xrCreateSecureMrPipelinePICO(m_secureMrFramework, &createInfo, &m_secureMrMap2dTo3dPipeline));

  // XrSecureMrPipelineTensorPICO uvPlaceholder;
  int32_t leftEyeUVDimensions[1] = {1};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, uvPlaceholder, leftEyeUVDimensions, 1, 2,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                    true);

  // XrSecureMrPipelineTensorPICO timestampPlaceholder;
  int32_t timestampDimensions[1] = {1};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, timestampPlaceholder, timestampDimensions, 1, 4,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_TIMESTAMP_PICO,
                                    true);

  // XrSecureMrPipelineTensorPICO cameraMatrixPlaceholder;
  int32_t cameraMatrixDimensions[2] = {3, 3};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, cameraMatrixPlaceholder, cameraMatrixDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    true);

  // XrSecureMrPipelineTensorPICO leftImgePlaceholder;
  int32_t leftImageDimensions[2] = {256, 256};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, leftImgePlaceholder, leftImageDimensions, 2, 3,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, true);

  // XrSecureMrPipelineTensorPICO rightImagePlaceholder;
  int32_t rightImageDimensions[2] = {256, 256};
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, rightImagePlaceholder, rightImageDimensions, 2, 3,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, true);

  // global Tensor
  int32_t positionDimensions[2] = {4, 4};
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, currentPosition, positionDimensions, 2, 1,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);
  m_apiHelper->CreateGlobalTensor(m_secureMrFramework, previousPosition, positionDimensions, 2, 1,
                                  XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, false);
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, currentPositionPlaceholder, positionDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    true);
  m_apiHelper->CreatePipelineTensor(m_secureMrMap2dTo3dPipeline, previousPositionPlaceholder, positionDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    true);

  // Map 2D landmark to 3D
  Map2Dto3D();
}

void FaceTrackingRaw::CreateSecureMrRenderingPipeline() {
  Log::Write(Log::Level::Info, "Open MR: CreateVSTImagePipeline");
  XrSecureMrPipelineCreateInfoPICO createInfo{XR_TYPE_SECURE_MR_PIPELINE_CREATE_INFO_PICO, nullptr};
  CHECK_XRCMD(xrCreateSecureMrPipelinePICO(m_secureMrFramework, &createInfo, &m_secureMrRenderingPipeline));

  Log::Write(Log::Level::Info, "Open MR: CreateRenderer");

  // set data to global tensors
  float tensorData[16] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  XrSecureMrTensorBufferPICO tensorBuffer{XR_TYPE_SECURE_MR_TENSOR_BUFFER_PICO, nullptr,
                                          static_cast<uint32_t>(sizeof(tensorData)), tensorData};
  CHECK_XRCMD(xrResetSecureMrTensorPICO(previousPosition, &tensorBuffer));
  CHECK_XRCMD(xrResetSecureMrTensorPICO(currentPosition, &tensorBuffer));

  // Initialize the previous inference result tensor
  int32_t previousInferenceResultDimensions[2] = {4, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrRenderingPipeline, previousRenderingPositionPlaceholder,
                                    previousInferenceResultDimensions, 2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, true);
  m_apiHelper->CreatePipelineTensor(m_secureMrRenderingPipeline, currentRenderingPositionPlaceholder,
                                    previousInferenceResultDimensions, 2, 1, XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO,
                                    XR_SECURE_MR_TENSOR_TYPE_MAT_PICO, true);

  //////////// interpolation
  XrSecureMrPipelineTensorPICO interpolatedResult;
  int32_t interpolatedResultDimensions[2] = {4, 4};
  m_apiHelper->CreatePipelineTensor(m_secureMrRenderingPipeline, interpolatedResult, interpolatedResultDimensions, 2, 1,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO, XR_SECURE_MR_TENSOR_TYPE_MAT_PICO,
                                    false);

  XrSecureMrOperatorPICO arithmetricOp;
  m_apiHelper->CreateArithmeticOperator(m_secureMrRenderingPipeline, arithmetricOp, "{0} * 0.95 + {1} * 0.05");
  m_apiHelper->SetInput(m_secureMrRenderingPipeline, arithmetricOp, previousRenderingPositionPlaceholder, "{0}");
  m_apiHelper->SetInput(m_secureMrRenderingPipeline, arithmetricOp, currentRenderingPositionPlaceholder, "{1}");
  m_apiHelper->SetOutput(m_secureMrRenderingPipeline, arithmetricOp, interpolatedResult, "result");

  // assign interpolated result to previous position
  XrSecureMrOperatorPICO assignmentOp;
  m_apiHelper->CreateOperator(m_secureMrRenderingPipeline, assignmentOp, XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO);
  m_apiHelper->SetInput(m_secureMrRenderingPipeline, assignmentOp, interpolatedResult, "src");
  m_apiHelper->SetOutput(m_secureMrRenderingPipeline, assignmentOp, previousRenderingPositionPlaceholder, "dst");

  // isVisible
  int32_t visibleDimensions[1] = {1};
  m_apiHelper->CreatePipelineTensor(m_secureMrRenderingPipeline, visiblePlaceholder, visibleDimensions, 1, 2,
                                    XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO, XR_SECURE_MR_TENSOR_TYPE_POINT_PICO,
                                    true);

  // load GLTF
  std::vector<char> gltfData;
  if (m_apiHelper->LoadModelData(GLTF_PATH, gltfData)) {
    XrSecureMrTensorCreateInfoGltfPICO gltfTensorCreateInfoPico{XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_GLTF_PICO, nullptr,
                                                                false, static_cast<uint32_t>(gltfData.size()),
                                                                gltfData.data()};
    CHECK_XRCMD(xrCreateSecureMrTensorPICO(
        m_secureMrFramework, reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO *>(&gltfTensorCreateInfoPico),
        &gltfAsset));
  } else {
    Log::Write(Log::Level::Error, "Failed to load GLTF data from file.");
  }
  // Create the gltf placeholder tensor
  XrSecureMrTensorCreateInfoGltfPICO gltfPlaceholderTensorCreateInfoPico{XR_TYPE_SECURE_MR_TENSOR_CREATE_INFO_GLTF_PICO,
                                                                         nullptr, true, 0, nullptr};
  CHECK_XRCMD(xrCreateSecureMrPipelineTensorPICO(
      m_secureMrRenderingPipeline,
      reinterpret_cast<XrSecureMrTensorCreateInfoBaseHeaderPICO *>(&gltfPlaceholderTensorCreateInfoPico),
      &gltfPlaceholderTensor));

  XrSecureMrOperatorPICO switchGltfOperator;
  XrSecureMrOperatorBaseHeaderPICO switchGltfOperatorInfo{XR_TYPE_SECURE_MR_OPERATOR_BASE_HEADER_PICO, nullptr};
  XrSecureMrOperatorCreateInfoPICO switchGltfOperatorCreateInfo{
      XR_TYPE_SECURE_MR_OPERATOR_CREATE_INFO_PICO, nullptr, &switchGltfOperatorInfo,
      XR_SECURE_MR_OPERATOR_TYPE_SWITCH_GLTF_RENDER_STATUS_PICO};
  CHECK_XRCMD(
      xrCreateSecureMrOperatorPICO(m_secureMrRenderingPipeline, &switchGltfOperatorCreateInfo, &switchGltfOperator));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrRenderingPipeline, switchGltfOperator,
                                                     gltfPlaceholderTensor, "gltf"));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrRenderingPipeline, switchGltfOperator,
                                                     interpolatedResult, "world pose"));
  CHECK_XRCMD(xrSetSecureMrOperatorOperandByNamePICO(m_secureMrRenderingPipeline, switchGltfOperator,
                                                     visiblePlaceholder, "visible"));
}

void FaceTrackingRaw::RunSecureMrVSTImagePipeline() {
  XrSecureMrPipelineIOPairPICO ioPairs[5];
  m_apiHelper->InitializePipelineIOPair(&ioPairs[0], vstTimestampPlaceholder, vstTimestamp);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[1], vstCameraMatrixPlaceholder, vstCameraMatrix);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[2], vstOutputLeftFp32Placeholder, vstOutputLeftFp32);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[3], vstOutputLeftUint8Placeholder, vstOutputLeftUint8);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[4], vstOutputRightUint8Placeholder, vstOutputRightUint8);

  XrSecureMrPipelineExecuteParameterPICO executeParams = {};
  executeParams.type = XR_TYPE_SECURE_MR_PIPELINE_EXECUTE_PARAMETER_PICO;
  executeParams.next = nullptr;
  executeParams.pipelineIOPair = ioPairs;
  executeParams.pairCount = sizeof(ioPairs) / sizeof(ioPairs[0]);

  XrSecureMrPipelineRunPICO pipelineRun;
  CHECK_XRCMD(xrExecuteSecureMrPipelinePICO(m_secureMrVSTImagePipeline, &executeParams, &pipelineRun));
}

void FaceTrackingRaw::RunSecureMrModelInferencePipeline() {
  XrSecureMrPipelineIOPairPICO ioPairs[3];
  m_apiHelper->InitializePipelineIOPair(&ioPairs[0], vstImagePlaceholder, vstOutputLeftFp32);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[1], leftEyeUVPlaceholder, leftEyeUVGlobal);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[2], isFaceDetectedPlaceholder, isFaceDetected);

  XrSecureMrPipelineExecuteParameterPICO executeParams = {};
  executeParams.type = XR_TYPE_SECURE_MR_PIPELINE_EXECUTE_PARAMETER_PICO;
  executeParams.next = nullptr;
  executeParams.pipelineIOPair = ioPairs;
  executeParams.pairCount = sizeof(ioPairs) / sizeof(ioPairs[0]);

  XrSecureMrPipelineRunPICO pipelineRun;
  CHECK_XRCMD(xrExecuteSecureMrPipelinePICO(m_secureMrModelInferencePipeline, &executeParams, &pipelineRun));
}

void FaceTrackingRaw::RunSecureMrMap2dTo3dPipeline() {
  XrSecureMrPipelineIOPairPICO ioPairs[7];
  m_apiHelper->InitializePipelineIOPair(&ioPairs[0], uvPlaceholder, leftEyeUVGlobal);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[1], timestampPlaceholder, vstTimestamp);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[2], cameraMatrixPlaceholder, vstCameraMatrix);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[3], leftImgePlaceholder, vstOutputLeftUint8);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[4], rightImagePlaceholder, vstOutputRightUint8);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[5], previousPositionPlaceholder, previousPosition);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[6], currentPositionPlaceholder, currentPosition);

  XrSecureMrPipelineExecuteParameterPICO executeParams = {};
  executeParams.type = XR_TYPE_SECURE_MR_PIPELINE_EXECUTE_PARAMETER_PICO;
  executeParams.next = nullptr;
  executeParams.pipelineIOPair = ioPairs;
  executeParams.pairCount = sizeof(ioPairs) / sizeof(ioPairs[0]);
  //            executeParams.conditionTensor = isFaceDetected;

  XrSecureMrPipelineRunPICO pipelineRun;
  CHECK_XRCMD(xrExecuteSecureMrPipelinePICO(m_secureMrMap2dTo3dPipeline, &executeParams, &pipelineRun));
}

void FaceTrackingRaw::RunSecureMrRenderingPipeline() {
  XrSecureMrPipelineIOPairPICO ioPairs[4];
  m_apiHelper->InitializePipelineIOPair(&ioPairs[0], gltfPlaceholderTensor, gltfAsset);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[1], visiblePlaceholder, isFaceDetected);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[2], previousRenderingPositionPlaceholder, previousPosition);
  m_apiHelper->InitializePipelineIOPair(&ioPairs[3], currentRenderingPositionPlaceholder, currentPosition);

  XrSecureMrPipelineExecuteParameterPICO executeParams = {};
  executeParams.type = XR_TYPE_SECURE_MR_PIPELINE_EXECUTE_PARAMETER_PICO;
  executeParams.next = nullptr;
  executeParams.pipelineIOPair = ioPairs;
  executeParams.pairCount = sizeof(ioPairs) / sizeof(ioPairs[0]);

  XrSecureMrPipelineRunPICO pipelineRun;
  CHECK_XRCMD(xrExecuteSecureMrPipelinePICO(m_secureMrRenderingPipeline, &executeParams, &pipelineRun));
}

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance &instance, const XrSession &session) {
  return std::make_shared<FaceTrackingRaw>(instance, session);
}
}  // namespace SecureMR
