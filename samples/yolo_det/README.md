## Sample: YOLO Object Detection

This sample builds an application for PICO using the SpatialML
APIs. The application demonstrates real-time object detection
using a YOLOX model converted from ONNX format.

![Demo of YOLO making detections on objects on the desk](../../docs/Demo-YOLO.gif)

### Code Walk-through

The implementation follows a pipeline-based architecture with four main stages:

1. **VST Image Pipeline**
   - Handles camera feed processing
   - Converts raw camera input to appropriate formats (uint8 and fp32)
   - Manages camera timestamps and matrices

2. **Model Inference Pipeline**
   - Processes the prepared images through the YOLO model
   - Performs class selection and NMS (Non-Maximum Suppression)
   - Outputs bounding boxes and confidence scores

3. **2D to 3D Mapping Pipeline**
   - Maps 2D detection results to 3D space
   - Uses stereo camera information for depth estimation
   - Generates 3D coordinates for detected objects

4. **Rendering Pipeline**
   - Visualizes detection results in the XR environment
   - Renders bounding boxes and labels using GLTF assets
   - Handles scale and positioning of visual elements

### Key Components

The yolo_object_detection.h class manages:

- **Framework Setup**: Initializes SpatialML runtime and pipeline components
- **Pipeline Management**: Creates and coordinates four main processing pipelines
- **Data Flow**: Uses GlobalTensors for inter-pipeline communication
- **Resource Management**: Handles GLTF assets and tensor memory
- **Threading**: Implements multi-threaded pipeline execution

### Configuration

- Model path: `yolo.serialized.bin`
- GLTF asset path: `frame2.gltf`
- Supports both left and right camera inputs for stereo processing

### Technical Details

The implementation uses SpatialML's pipeline architecture for efficient:
- Tensor operations
- Multi-threaded execution
- Hardware-accelerated processing
- XR session management
- Real-time visualization

Each pipeline can be monitored and controlled independently, with synchronized data flow managed through global tensors.
