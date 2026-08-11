## Sample: UFO (Raw usage of C-API)

_The sample achieves exactly the same effect as_
_the one in `${PROJ_ROOT}/samples/ufo`. However,_
_the sample application is re-written using the_
_raw C-API from the OpenXR extension for SecureMR,_
_without usage of the utility classes provided in_
_`${PROJ_ROOT}/external/SpatialML-XR-Utils/base/securemr_utils` to demonstrate:_

1. _the most fundermental usage of the raw OpenXR extension, and_
1. _the convenience that the utility classes provide_

### Visual effect

This sample builds an application for PICO using the
SecureMR APIs. In the application, users become a 
pilot who remotely control a disc-shape UFO to chase
human beings whoever they see. By simply looking 
at other people's heads, the user will find their UFO
flys towards the detected human being and floats
above their head. 

The sample deploys an open-sourced face detection from
MediaPipeline via SecureMR, and renders a UFO of
glTF 2.0 format. 

### Code walk-through

_**TODO**_
