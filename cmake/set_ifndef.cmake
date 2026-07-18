# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2024-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2024-04-19 14:34:02
#  **************************************************************/

function (set_ifndef variable value)
  if(NOT DEFINED ${variable})
    set(${variable} ${value} PARENT_SCOPE)
  endif()
  
endfunction()
