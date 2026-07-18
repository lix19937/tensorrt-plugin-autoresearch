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

# KERNEL=
# -k ${KERNEL}                              \

${NCU} -f        \
--set full       \
--launch-skip 50 \
--launch-count 2 \
--target-processes all                    \
-o ${WORKSPACES}/exp                      \
${BIN_FILE} ${JSON_FILE} ${PATTERN} ${GROUP_IDX} > /dev/null

${NCU} -i ${WORKSPACES}/exp.ncu-rep --log-file ${WORKSPACES}/exp.log

echo -e "\n\n -------------------- ncu profile information -------------------- "

${NCU}  -i ${WORKSPACES}/exp.ncu-rep  --print-summary per-kernel

