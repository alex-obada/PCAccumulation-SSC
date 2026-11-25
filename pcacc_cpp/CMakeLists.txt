cmake_minimum_required(VERSION 3.22)
project(pcacc LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # compile_commands.json for VS Code IntelliSense
# nvcc gets its -I flags from a .rsp file by default, which cpptools can't read: inline them
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87)          # Jetson; on the laptop: -DCMAKE_CUDA_ARCHITECTURES=89
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# TensorRT: headers + libs from the NVIDIA tar archive on the laptop, /usr on the Jetson.
# Until it is installed the project still builds (CUDA modules + tests); the engine stages are skipped.
set(TENSORRT_ROOT "" CACHE PATH "TensorRT path (empty on the Jetson, where it is in /usr)")
find_path(TRT_INCLUDE NvInfer.h HINTS ${TENSORRT_ROOT}/include /usr/include/aarch64-linux-gnu)
find_library(NVINFER nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSER nvonnxparser HINTS ${TENSORRT_ROOT}/lib)
if(TRT_INCLUDE AND NVINFER AND NVONNXPARSER)
  set(PCACC_WITH_TENSORRT ON)
  message(STATUS "TensorRT: ${TRT_INCLUDE}")
else()
  set(PCACC_WITH_TENSORRT OFF)
  message(WARNING "TensorRT not found (set -DTENSORRT_ROOT=...): building without the engine stages")
endif()

# ---------------------------------------------------------------- core library (everything but main)
file(GLOB CORE_SRC CONFIGURE_DEPENDS src/*.cpp src/cuda/*.cu)
list(REMOVE_ITEM CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
if(NOT PCACC_WITH_TENSORRT)
  list(REMOVE_ITEM CORE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/trt_engine.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/networks.cpp)
endif()

add_library(pcacc_core STATIC ${CORE_SRC})
target_include_directories(pcacc_core PUBLIC include)
target_link_libraries(pcacc_core PUBLIC CUDA::cudart Eigen3::Eigen)
target_compile_options(pcacc_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
if(PCACC_WITH_TENSORRT)
  target_include_directories(pcacc_core SYSTEM PUBLIC ${TRT_INCLUDE})   # no warnings from NVIDIA headers
  target_link_libraries(pcacc_core PUBLIC ${NVINFER} ${NVONNXPARSER})
  target_compile_definitions(pcacc_core PUBLIC PCACC_WITH_TENSORRT=1)
endif()

add_executable(pcacc src/main.cpp)
target_link_libraries(pcacc PRIVATE pcacc_core)

# ---------------------------------------------------------------- tests (reference outputs in data/)
enable_testing()
add_executable(test_modules tests/test_modules.cpp)
target_link_libraries(test_modules PRIVATE pcacc_core)
set(PCACC_TEST_DATA ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "frames + reference outputs")
set(PCACC_MODELS ${CMAKE_CURRENT_SOURCE_DIR}/models CACHE PATH ".onnx models + cached engines")
set(PCACC_TESTS preprocess voxelize pillar_mean point_features pillar_max_concat bev)
if(PCACC_WITH_TENSORRT)
  list(APPEND PCACC_TESTS pfn_engines pillar_encoder backbone)
endif()
foreach(t ${PCACC_TESTS})
  add_test(NAME ${t} COMMAND test_modules ${t} ${PCACC_TEST_DATA} ${PCACC_MODELS})
endforeach()
if(PCACC_WITH_TENSORRT)
  # the first run builds the engines (minutes); later runs load them from models/*.engine
  set_tests_properties(pfn_engines pillar_encoder backbone PROPERTIES TIMEOUT 1800 RUN_SERIAL TRUE)
endif()
