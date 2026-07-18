# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2026-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2026-04-21 14:34:02
#  **************************************************************/
#!/bin/bash

SHELL_PATH=$(cd $(dirname "$0") && pwd)

WORKSPACES=${SHELL_PATH}/../../../..

TARGET=pillarvfe

BIN_FILE=${WORKSPACES}/build/t_${TARGET}
JSON_FILE=${WORKSPACES}/src/${TARGET}/config.json
GROUP_IDX=0

NSYS=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu

