cmake_minimum_required(VERSION 3.22)
project(model_inspect)

set(SAMPLE_SRCS ${CMAKE_CURRENT_LIST_DIR}/cpp/model_inspect.cpp)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)

set(SPATIALML_XR_UTILS_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../external/SpatialML-XR-Utils"
    CACHE PATH "Path to SpatialML-XR-Utils checkout")
include("${SPATIALML_XR_UTILS_DIR}/cmake/SpatialMLXRUtils.cmake")

spatialml_xr_project(
    TARGET ${PROJECT_NAME}
    SOURCES ${SAMPLE_SRCS}
    INCLUDE_DIRS
        ${SAMPLE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${THIRD_PARTY_DIRS}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
