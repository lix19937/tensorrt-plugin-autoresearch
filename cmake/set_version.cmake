# /**************************************************************
#  * @Author: ljw 
#  * @Date: 2024-04-19 10:14:11 
#  * @Last Modified by: ljw 
#  * @Last Modified time: 2024-04-19 14:34:02
#  **************************************************************/

##
## ref  https://semver.org/lang/zh-CN/
##
## format 1.0.0-alpha、1.0.0-alpha.1、1.0.0-0.3.7、1.0.0-x.7.z.92
##        1.0.0-alpha+001、1.0.0-20130313144700+des、1.0.0-beta+exp.sha.5114f85
##        1.9.1、1.10.0、1.11.0、2.3.1、4.6.8
##
## Ensure uniform format，here we use '-' as a connector, refer to https://github.com/pytorch/pytorch 
## ref https://cloud.tencent.com/developer/ask/sof/44655
##
## After you get valid version, then 
## set_target_properties(${LIB_NAME} PROPERTIES VERSION ${OPS_VERSION} SOVERSION ${OPS_SOVERSION} )
##
function(check len gt_len vstr)
  if (${len} STREQUAL ${gt_len})
    message(STATUS "VERSION_STR:${vstr}")
  else()
    message(FATAL_ERROR "Bad format:${vstr}")
  endif()
endfunction()

function(checknone src vstr)
  if(${src} MATCHES "^$")
    message(FATAL_ERROR "Bad format:${vstr}")
  endif()
endfunction()

function (set_version __file __version __soversion)
  execute_process(
    COMMAND head -1 ${__file}  RESULT_VARIABLE ret OUTPUT_VARIABLE version
  )
  set(VERSION_STR ${version})
  string(STRIP ${VERSION_STR} VERSION_STR_STRIP)  

  string(FIND ${VERSION_STR_STRIP} " " SPACE_FLAG)
  if (${SPACE_FLAG} STREQUAL -1)
  else()
    message(FATAL_ERROR "Bad format, has space:" ${SPACE_FLAG},${VERSION_STR_STRIP} ) 
  endif()

  string(REPLACE "\n" ";" VERSION_STR_PURE_LIST ${VERSION_STR_STRIP}) # SPLIT 1th
  list(GET VERSION_STR_PURE_LIST 0 VERSION_STR_PURE)

  string(REPLACE "-" ";" VERSION_LIST ${VERSION_STR_PURE}) # SPLIT 2th
  list(LENGTH VERSION_LIST len)

  set(RV OFF)

  if (${len} STREQUAL 2)
    list(GET VERSION_LIST 0 VERSION_MAJOR_MIN_PATCH)
    list(GET VERSION_LIST 1 VERSION_PRE_RELEASE)

    string(REPLACE "." ";" VERSION_LIST ${VERSION_MAJOR_MIN_PATCH}) # SPLIT 3th

    list(LENGTH VERSION_LIST len)
    check(${len} 3 ${VERSION_STR_STRIP})

    list(GET VERSION_LIST 0 VERSION_MAJOR)
    list(GET VERSION_LIST 1 VERSION_MINOR)
    checknone(VERSION_MINOR ${VERSION_STR_STRIP})
    list(GET VERSION_LIST 2 VERSION_PATCH)
    checknone(VERSION_PATCH ${VERSION_STR_STRIP})

  elseif(${len} STREQUAL 1)
    set(RV ON)
    string(REPLACE "." ";" VERSION_LIST ${VERSION_STR_STRIP}) # SPLIT 2th

    list(LENGTH VERSION_LIST len)
    check(${len} 3 ${VERSION_STR_STRIP})

    list(GET VERSION_LIST 0 VERSION_MAJOR)
    list(GET VERSION_LIST 1 VERSION_MINOR)
    checknone(VERSION_MINOR ${VERSION_STR_STRIP})
    list(GET VERSION_LIST 2 VERSION_PATCH)
    checknone(VERSION_PATCH ${VERSION_STR_STRIP})
  endif()

  set(_MAJOR ${VERSION_MAJOR})
  set(_MINOR ${VERSION_MINOR})
  set(_PATCH ${VERSION_PATCH})
  set(_PR    ${VERSION_PRE_RELEASE})

  if(RV)
    set(${__version} "${_MAJOR}.${_MINOR}.${_PATCH}" CACHE STRING "project version")
  else()
    set(${__version} "${_MAJOR}.${_MINOR}.${_PATCH}-${_PR}" CACHE STRING "project version")
  endif()

  set(${__soversion} "${_MAJOR}" CACHE STRING "library so version")
endfunction()
