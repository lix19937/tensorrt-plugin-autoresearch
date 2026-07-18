# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2024-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2024-04-19 14:34:02
#  **************************************************************/

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
  add_definitions (-ggdb -Wall -D_GNU_SOURCE=1 -D__STDC_LIMIT_MACROS=1)
  message(STATUS "C++98 activated.")
  message(FATAL_ERROR "compile exit, CMAKE_CXX_COMPILER_VERSION (${CMAKE_CXX_COMPILER_VERSION}) < 4.7")
endif()

message(STATUS "C++17 activated.")
message(STATUS "Build type:${CMAKE_BUILD_TYPE}")
message(STATUS "Build cxx flags:${CMAKE_CXX_FLAGS}")
message(STATUS "Debug configuration:${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "Release configuration:${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "Release configuration with debug info:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "Minimal release configuration:${CMAKE_CXX_FLAGS_MINSIZEREL}")

set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CXX_STD}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -fpic -fpie")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type -fopenmp")
set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}   -DWMMA")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DWMMA")
set(CMAKE_CUDA_STANDARD 17)

if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  endif()

  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.1)
    list(APPEND GPU_ARCHS 86)
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()

# Generate SASS for each architecture
foreach(arch ${GPU_ARCHS})
  set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
  set(ENABLED_SMS "${ENABLED_SMS} -DENABLE_SM${arch}")
endforeach()

# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")

# Set CMAKE_CUDA_FLAGS
if(NOT MSVC)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Xcompiler -fPIC -Wno-deprecated-declarations")

else()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler")
endif()

#  Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 86 87 90)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

# set(ENABLED_SMS "-DENABLE_SM80 -DENABLE_SM86 -DENABLE_SM87 -DENABLE_SM89")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ENABLED_SMS}")

set(LINE_FEED " ")
set(DEBUG_POSTFIX _debug CACHE STRING "suffix for debug builds")
message(${LINE_FEED})

if (CMAKE_BUILD_TYPE STREQUAL "debug")
  message(STATUS "Building in debug mode ${DEBUG_POSTFIX}")
endif()

if (${CMAKE_BUILD_TYPE} MATCHES "debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG") 
endif()

message(STATUS ${CMAKE_CXX_FLAGS})
message(STATUS ${CMAKE_CUDA_FLAGS})

message(${LINE_FEED})
