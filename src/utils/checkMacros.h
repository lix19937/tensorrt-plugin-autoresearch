/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "stringUtils.h"

namespace trt_edgellm {
namespace check {

inline void check(bool condition, std::string errorMsg) {
  if (!condition) {
    printf("Error, %s\n", errorMsg.c_str());
    throw std::runtime_error(errorMsg);
  }
}

inline void _checkCuda(
  cudaError_t result,
  char const* const func,
  [[maybe_unused]] char const* const file,
  [[maybe_unused]] int const line) {
  if (result) {
    throw std::runtime_error(format::fmtstr("CUDA runtime error in %s: %s", func, cudaGetErrorString(result)));
  }
}

inline void _checkCudaDriver(
  CUresult result, char const* const func, [[maybe_unused]] char const* const file, [[maybe_unused]] int const line) {
  if (result) {
    char const* errorName = nullptr;
    if (cuGetErrorName(result, &errorName) != CUDA_SUCCESS) {
      errorName = "CUDA driver API error happened, but we failed to get error name.";
    }
    throw std::runtime_error(format::fmtstr("CUDA driver API error in %s: %s", func, errorName));
  }
}

} // namespace check

#define CHECK(condition)              \
  do {                                \
    check((condition), "check error") \
  } while (0)

#define CUDA_CHECK(stat)                                               \
  do {                                                                 \
    trt_edgellm::check::_checkCuda((stat), #stat, __FILE__, __LINE__); \
  } while (0)

#define CUDA_DRIVER_CHECK(stat)                                              \
  do {                                                                       \
    trt_edgellm::check::_checkCudaDriver((stat), #stat, __FILE__, __LINE__); \
  } while (0)

} // namespace trt_edgellm
