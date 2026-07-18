/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#pragma once

#include <cuda.h>

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 11080)
#include <cuda_fp8.h>
#define SUPPORTS_FP8 1
#else
#define SUPPORTS_FP8 0
#endif

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#include <cuda_fp4.h>
#define SUPPORTS_FP4 1
#else
#define SUPPORTS_FP4 0
#endif
