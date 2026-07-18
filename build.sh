# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2026-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2026-04-21 14:34:02
#  **************************************************************/
#!/bin/bash

set -e

show_help() {
    echo "Usage: $0 [thoru|x86]"
    echo ""
    echo "option:"
    echo "  -h, --help"
    echo "  -v, --version"
    echo ""
    echo "eg:"
    echo "  $0 thoru"
}

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
    exit 0
fi

SHELL_PATH=$(cd $(dirname "$0") && pwd)

BASE_PATH=$SHELL_PATH/../env

CUDA_VERSION=12.8
ARCH=aarch64
NV_SDK_PATH=$BASE_PATH/nv_sdk/thoru-aarch64
HOST_NVCC_PATH=$BASE_PATH/nv_sdk/thoru-x86_64/cuda/usr/local/cuda-${CUDA_VERSION}/bin

if [ $# -ge 1 ] && [[ "$1" =~ "thoru" ]];then
    echo "-- THORU build"

    NV_SDK_PATH=$BASE_PATH/nv_sdk/thoru-aarch64
    ARCH=aarch64
else 
    echo "-- X86 build"

    ARCH=x86_64
    NV_SDK_PATH=$BASE_PATH/nv_sdk/thoru-x86_64
fi

export PATH=$HOST_NVCC_PATH:$PATH
echo "-- Check for working CUDA compiler: $(which nvcc)"

dirname=$SHELL_PATH/build
if [ ! -d ${dirname} ]; then
  mkdir -m 777 ${dirname}
fi

rm -rf ${dirname}/*  &&   

cd ${dirname}

mycmake=cmake #/workspaces/thoru/gw00348951/auto-cuda-optimizer/cmake-3.30.5-linux-x86_64/bin/cmake

if [ $# -ge 1 ] && [[ "$1" =~ "thoru" ]];then
  $mycmake ..                                      \
  -D CMAKE_BUILD_TYPE=release                      \
  -D RUN_PLATFORM=$ARCH                            \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache              \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache            \
  -D CMAKE_TOOLCHAIN_FILE=../cmake/toolchain.cmake \
  -D NV_SDK_PATH=$NV_SDK_PATH                      \
  -D CUDA_VERSION=$CUDA_VERSION                    \
  -D HOST_NVCC_PATH=$HOST_NVCC_PATH                \
  -D CMAKE_CUDA_ARCHITECTURES=101                  \
  -D GPU_ARCHS=110                                 

else
  $mycmake ..                                      \
  -D CMAKE_BUILD_TYPE=release                      \
  -D RUN_PLATFORM=$ARCH                            \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache              \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache            \
  -D NV_SDK_PATH=$NV_SDK_PATH                      \
  -D CUDA_VERSION=$CUDA_VERSION                    \
  -D CMAKE_CUDA_ARCHITECTURES=89                   \
  -D GPU_ARCHS=89                                 

fi

make -j12

# rm -fr /workspaces/qm_rsp_model/qm_model/arch64_0915/libcc_trt_plugin.so

# cp  -fr  ${dirname}/libcc_trt_plugin.so     /workspaces/qm_rsp_model/qm_model/arch64_0915/
