# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2026-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2026-04-19 14:34:02
#  **************************************************************/

set(CURR_DIR ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(CURR_DIR_NAME ${CURR_DIR} NAME)

message(STATUS "CURR_DIR:${CURR_DIR_NAME}")

# ------------------------------------
FILE(GLOB SRC_LIB_BASELINE *.cu  *.cpp  ../utils/*.cpp)
set(LIB_BASELINE baseline)
add_library(${LIB_BASELINE} SHARED
  ${SRC_LIB_BASELINE}
)
target_link_libraries(${LIB_BASELINE} cudart nvinfer)
target_include_directories(${LIB_BASELINE} PUBLIC ../utils)  

# ------------------------------------
FILE(GLOB SRC_LIB_EXP  exp/*.cu  *.cpp  ../utils/*.cpp)
set(LIB_EXP exp)
add_library(${LIB_EXP} SHARED
  ${SRC_LIB_EXP}
)
target_link_libraries(${LIB_EXP} cudart nvinfer)
target_include_directories(${LIB_EXP} PUBLIC ../utils ./)  

# ------------------------------------
set(BIN_NAME t_${CURR_DIR_NAME})
FILE(GLOB SRC_PIPELINE  ../main.cpp  ../utils/*.cpp)
add_executable(${BIN_NAME} ${SRC_PIPELINE})
target_link_libraries(${BIN_NAME} cudart  cuda)
target_include_directories(${BIN_NAME} PUBLIC ../utils  ./ )  
