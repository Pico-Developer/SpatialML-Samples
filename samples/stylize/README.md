## Sample: Stylize Demo

This sample builds an application for PICO using the
SpatialML APIs. The application demonstrates neural
style transfer on live VST imagery and renders the
stylized output onto two portal surfaces.

The sample deploys a style predictor model and a
style transfer model through SpatialML, and maps
the output to a glTF portal mesh.

### Render

The render asset for this sample is a glTF portal
mesh. Each portal is rendered as an independent
instance whose base color texture is updated at
runtime with the stylized output.
This sample demonstrates dynamic texture tensors,
which are used on the portal glTFs, allowing the
texture contents to be updated directly without
issuing a texture update render command.

Two portals are placed in front of the user, and
their poses are updated from controller input to
drive the VST crop that feeds the style transfer
model.

### Code walk-through

There are four pipelines routinely used in this
sample, plus one one-time initialization pipeline.

1. The VST pipeline, `m_secureMrVSTImagePipeline`,
   queries the camera for the left-eye image and
   stores it in the global tensor
   `vstOutputLeftUint8Global` at 2048x1536 uint8.
2. The style prediction pipeline,
   `m_secureMrStylePredictionPipeline`, runs the
   style predictor model on each selected style
   texture (256x256) and stores the 100-channel
   style embedding into `predictedStyleGlobal`.
   This pipeline is re-run only when a new style
   texture is chosen.
3. The style transfer pipeline,
   `m_secureMrStyleTransferPipeline`, computes an
   affine transform from portal screen-space
   source points (`srcPointsGlobal`), slices the
   VST image to 384x384, runs the style transfer
   model, and writes the stylized output into the
   GPU dynamic texture tensor
   `stylizedImageGPUGlobal`.
4. The rendering pipeline,
   `m_secureMrRenderingPipeline`, renders the
   portal glTF assets using per-portal pose
   matrices from `portalPoseGlobal`.
5. The one-time initialization pipeline inside
   `CreateGlobalTensor` loads the portal glTF,
   initializes portal poses, and binds each portal
   material to the corresponding stylized texture.

Controller pose updates drive portal positioning
and the screen-space source points used by the
affine transform. Button presses advance the style
texture for the left portal, right portal, or both.
