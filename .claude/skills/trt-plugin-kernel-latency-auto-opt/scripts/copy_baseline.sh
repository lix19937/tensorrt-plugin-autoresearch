# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2026-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2026-04-21 14:34:02
#  **************************************************************/
#!/bin/bash

set -e 

SHELL_PATH=$(cd $(dirname "$0") && pwd)

source ${SHELL_PATH}/common.sh

BASELINE_PATH=${WORKSPACES}/src/${TARGET}
EXP_DIR=${WORKSPACES}/src/${TARGET}/exp

if [ ! -d ${EXP_DIR} ]; then
  mkdir -m 777 ${EXP_DIR}
  cp -fr ${BASELINE_PATH}/*.cu ${EXP_DIR}/
else
  echo "--"
fi

