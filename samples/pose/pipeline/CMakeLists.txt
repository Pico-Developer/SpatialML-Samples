cmake_minimum_required(VERSION 3.22)
project(pose_pipeline)

set(SAMPLE_SRCS "${CMAKE_CURRENT_LIST_DIR}/../../pipeline_common/pipeline_package_app.cpp")
set(USE_SECURE_MR_UTILS ON)
set(SPATIALML_XR_UTILS_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../external/SpatialML-XR-Utils"
    CACHE PATH "Path to SpatialML-XR-Utils checkout")
include("${SPATIALML_XR_UTILS_DIR}/cmake/SpatialMLXRUtils.cmake")

spatialml_xr_project(
    TARGET ${PROJECT_NAME}
    SOURCES ${SAMPLE_SRCS}
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/../../pipeline_common
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    PIPELINE_PACKAGE_ROOT="pose-pipeline"
    PIPELINE_IDS="inference,display"
    PIPELINE_APP_TAG="Pose Pipeline"
    PIPELINE_FRAME_WIDTH=512
    PIPELINE_FRAME_HEIGHT=512
)

