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

PATTERN=1

${NSYS} profile                                    \
--stats=true -f true                               \
-o ${WORKSPACES}/exp.nsys-rep                      \
${BIN_FILE} ${JSON_FILE} ${PATTERN} ${GROUP_IDX} 

