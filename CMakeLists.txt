# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2024-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2024-04-19 14:34:02
#  **************************************************************/

# only support ubuntu >= 18.04. which has cmake 3.10
# the reason why we need it is because FindCUDA/find_package(CUDA)
# is depricated since cmake 3.10.
# refer to https://cmake.org/cmake/help/v3.25/module/FindCUDA.html
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(
  nni 
  LANGUAGES CXX CUDA
  DESCRIPTION "nni"
  HOMEPAGE_URL "https://github.com/lix19937")

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(cmake/set_ifndef.cmake)
include(cmake/set_version.cmake)
include(cmake/options.cmake)

set_ifndef(GPU_ARCHS        90)
set_ifndef(CMAKE_BUILD_TYPE release) 
set_ifndef(RUN_PLATFORM     aarch64)

include(cmake/basic.cmake)

if (${RUN_PLATFORM} STREQUAL "x86_64")
  message(STATUS "AMD64 platform")
  link_directories(${NV_SDK_PATH}/cudnn/usr/lib/x86_64-linux-gnu)
  link_directories(${NV_SDK_PATH}/cudnn/usr/lib/x86_64-linux-gnu/stubs)

  link_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib)
  link_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/stubs)

  include_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/include)   
  include_directories(${NV_SDK_PATH}/cudnn/usr/include/x86_64-linux-gnu)   

  link_directories(${NV_SDK_PATH}/tensorrt/usr/lib/x86_64-linux-gnu)
  link_directories(${NV_SDK_PATH}/tensorrt/usr/lib/x86_64-linux-gnu/stubs)
  include_directories(${NV_SDK_PATH}/tensorrt/usr/include/x86_64-linux-gnu)
  
  add_definitions(-D TARGET_x86_64)  
  set(NVBench_CUPTI_LIBRARY ${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/x86_64-linux/lib/libcupti.so)

else()
  link_directories(${NV_SDK_PATH}/cudnn/usr/lib/aarch64-linux-gnu)
  link_directories(${NV_SDK_PATH}/cudnn/usr/lib/aarch64-linux-gnu/stubs)

  link_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/lib)
  link_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/lib/stubs)
  include_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/include)   

  include_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/thor/targets/aarch64-linux/include)   
  link_directories(${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/thor/targets/aarch64-linux/lib/stubs)

  link_directories(${NV_SDK_PATH}/tensorrt/usr/lib/aarch64-linux-gnu)
  link_directories(${NV_SDK_PATH}/tensorrt/usr/lib/aarch64-linux-gnu/stubs)
  include_directories(${NV_SDK_PATH}/tensorrt/usr/include/aarch64-linux-gnu)  

  set(NVBench_CUPTI_LIBRARY ${NV_SDK_PATH}/cuda/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/lib/libcupti.so)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)  

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH    ${PROJECT_SOURCE_DIR}/build)

message(STATUS ${RUN_PLATFORM})

add_definitions(-D PARKING_MT)  
add_definitions(-D __DEFINE_INDEPEND_TEST__)  

add_subdirectory(src)
