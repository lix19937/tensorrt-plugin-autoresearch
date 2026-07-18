# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2024-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2024-04-19 14:34:02
#  **************************************************************/

SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR aarch64)

SET(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
SET(CMAKE_C_COMPILER   /usr/bin/aarch64-linux-gnu-gcc)

SET(CMAKE_FIND_ROOT_PATH ${NV_SDK_PATH})

# https://github.com/bootlin/buildroot-toolchains/tree/toolchains.bootlin.com-2024.02
# SET(CMAKE_LIBRARY_PATH 
#   ${CUDAX_ENV_PATH}/usr/lib/aarch64-linux-gnu)   # cudnn nvinfer 
# SET(CMAKE_INCLUDE_PATH 
#   ${CUDAX_ENV_PATH}/usr/include/aarch64-linux-gnu # cudnn nvinfer inc 
#   ${CUDAX_ENV_PATH}/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/include) # cuda inc 
# SET(LD_LIBRARY_PATH 
#   ${NVX_ENV_PATH}/filesystem/targetfs/usr/lib/aarch64-linux-gnu
#   ${CUDAX_ENV_PATH}/usr/lib/aarch64-linux-gnu  # cudnn nvinfer 
#   ${CUDAX_ENV_PATH}/usr/local/cuda-${CUDA_VERSION}/targets/aarch64-linux/lib/stubs) # cublas cudla

# Have to set this one to BOTH, to allow CMake to find rospack
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Use host nvcc
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_COMPILER ${HOST_NVCC_PATH}/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_FORCED TRUE)
