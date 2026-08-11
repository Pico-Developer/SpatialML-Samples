cmake_minimum_required(VERSION 3.22)
project(rubics_cube)

set(SAMPLE_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/cpp/rubics_cube.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/camera_manager.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/color_classifier.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/scan_ui.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/scan_manager.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/cube_model.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/solver.cpp
        ${CMAKE_CURRENT_LIST_DIR}/cpp/virtual_cube.cpp
)
set(SAMPLE_DIR ${CMAKE_CURRENT_LIST_DIR}/cpp)
set(USE_SECURE_MR_UTILS ON)
set(THIRD_PARTY_DIRS ${CMAKE_CURRENT_LIST_DIR}/third_party)
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/kociemba.cmake)

# OpenCV (Android only)
if(ANDROID)
    # This creates an imported target named `lib_opencv` and sets include dirs.
    include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/opencv.cmake)
endif()

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

# Link OpenCV to the current sample target (Android only)
if(ANDROID)
    target_link_libraries(${PROJECT_NAME} PRIVATE lib_opencv)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE kociemba_c)
