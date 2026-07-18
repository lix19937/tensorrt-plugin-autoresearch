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

PATTERN=2

if [ -z "$1" ]; then
    echo "Error: Missing required argument!"
    echo -e "Usage: $0 <argument>\n"
    exit 1
fi

${BIN_FILE} ${JSON_FILE} ${PATTERN} $1 ${GROUP_IDX}
