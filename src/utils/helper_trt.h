/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#define DRIVEOS_7030 1

#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#ifdef DRIVEOS_7030
#include <cuda_fp8.h>
#endif

#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include "helper_bin.h"

namespace utils {

inline int64_t GetNumel(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, &d.d[d.nbDims], 1, std::multiplies<int>{});
}

inline int64_t GetNbBytes(nvinfer1::DataType t, int64_t numel) {
  switch (t) {
#ifdef DRIVEOS_6060
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4 * numel;
    case nvinfer1::DataType::kHALF:
      return 2 * numel;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
      return numel;
#else
    case nvinfer1::DataType::kINT64:
      return 8 * numel;
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4 * numel;
    case nvinfer1::DataType::kBF16:
    case nvinfer1::DataType::kHALF:
      return 2 * numel;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
      return numel;

    case nvinfer1::DataType::kFP8:
#if CUDA_VERSION < 11060
      printf("FP8 is not supported");
#else
      return numel;
#endif

    case nvinfer1::DataType::kINT4:
    case nvinfer1::DataType::kFP4:
      return (numel + 1) / 2;
#endif
  }

  printf("Unknown element type %d", int(t));
  return 0;
}

inline float GetDiffThresh(nvinfer1::DataType dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT)
    return 1e-6f;
  if (dtype == nvinfer1::DataType::kHALF)
    return 1e-3f;
  if (dtype == nvinfer1::DataType::kINT32)
    return 0.f;
  if (dtype == nvinfer1::DataType::kINT64)
    return 0.f;

  return 1e-3f;
}

inline int64_t GetNbBytes(const nvinfer1::PluginTensorDesc& desc) {
  return GetNbBytes(desc.type, GetNumel(desc.dims));
}

inline std::string GetTensor(const void* input, nvinfer1::Dims dims, nvinfer1::DataType dtype, int limit = 32) {
  int num = int(GetNumel(dims));
  if (num <= 0)
    return "";

  if (num > 64)
    return "";

  if (num > limit)
    num = limit;

  std::stringstream ss;
  ss << "[";
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT: {
      std::vector<float> buffer(num);
      CheckCudaErrors(cudaMemcpy(buffer.data(), input, num * sizeof(float), cudaMemcpyDefault));
      for (int i = 0; i < num; ++i) {
        ss << buffer[i];
        if (i < num - 1) {
          ss << ", ";
        }
      }
    } break;

    case nvinfer1::DataType::kHALF: {
      std::vector<half> buffer(num);
      CheckCudaErrors(cudaMemcpy(buffer.data(), input, num * sizeof(half), cudaMemcpyDefault));
      for (int i = 0; i < num; ++i) {
        ss << __half2float(buffer[i]);
        if (i < num - 1) {
          ss << ", ";
        }
      }
    } break;

    case nvinfer1::DataType::kINT32: {
      std::vector<int32_t> buffer(num);
      CheckCudaErrors(cudaMemcpy(buffer.data(), input, num * sizeof(int32_t), cudaMemcpyDefault));
      for (int i = 0; i < num; ++i) {
        ss << buffer[i];
        if (i < num - 1) {
          ss << ", ";
        }
      }
    } break;

    case nvinfer1::DataType::kINT64: {
      std::vector<int64_t> buffer(num);
      CheckCudaErrors(cudaMemcpy(buffer.data(), input, num * sizeof(int64_t), cudaMemcpyDefault));
      for (int i = 0; i < num; ++i) {
        ss << buffer[i];
        if (i < num - 1) {
          ss << ", ";
        }
      }
    } break;

    case nvinfer1::DataType::kINT8: {
      std::vector<int64_t> buffer(num);
      CheckCudaErrors(cudaMemcpy(buffer.data(), input, num * sizeof(int8_t), cudaMemcpyDefault));
      for (int i = 0; i < num; ++i) {
        ss << int(buffer[i]);
        if (i < num - 1) {
          ss << ", ";
        }
      }
    } break;

    default:
      break;
  }
  ss << "]";

  return ss.str();
}

inline void GetDesc(
  const nvinfer1::PluginTensorDesc* in_desc,
  const nvinfer1::PluginTensorDesc* out_desc,
  const void* const* inputs,
  int num_in,
  int num_out,
  int limit) {
  auto fmt_dims = [](nvinfer1::Dims dims) {
    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
      ss << dims.d[i];
      if (i < dims.nbDims - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  };

  auto fmt_dtype = [](nvinfer1::DataType dtype) {
    std::string type{"unknown"};
    switch (dtype) {
      case nvinfer1::DataType::kHALF:
        type = "fp16";
        break;
      case nvinfer1::DataType::kFLOAT:
        type = "fp32";
        break;
      case nvinfer1::DataType::kINT32:
        type = "int32";
        break;
      case nvinfer1::DataType::kINT64:
        type = "int64";
        break;
      case nvinfer1::DataType::kINT8:
        type = "int8";
        break;
      case nvinfer1::DataType::kFP8:
        type = "fp8";
        break;

      default:
        break;
    }

    return type;
  };

  auto fmt_layout = [](nvinfer1::TensorFormat fmt) {
    std::string layfmt{"unknown"};
    switch (fmt) {
      case nvinfer1::TensorFormat::kLINEAR:
        layfmt = "kLINEAR";
        break;
      case nvinfer1::TensorFormat::kCHW2:
        layfmt = "kCHW2";
        break;
      case nvinfer1::TensorFormat::kHWC8:
        layfmt = "kHWC8";
        break;
      case nvinfer1::TensorFormat::kCHW4:
        layfmt = "kCHW4";
        break;
      case nvinfer1::TensorFormat::kCHW16:
        layfmt = "kCHW16";
        break;
      case nvinfer1::TensorFormat::kCHW32:
        layfmt = "kCHW32";
        break;
      case nvinfer1::TensorFormat::kDHWC8:
        layfmt = "kDHWC8";
        break;
      case nvinfer1::TensorFormat::kCDHW32:
        layfmt = "kCDHW32";
        break;
      case nvinfer1::TensorFormat::kHWC:
        layfmt = "kHWC";
        break;
      case nvinfer1::TensorFormat::kDLA_LINEAR:
        layfmt = "kDLA_LINEAR";
        break;
      case nvinfer1::TensorFormat::kDLA_HWC4:
        layfmt = "kDLA_HWC4";
        break;
      case nvinfer1::TensorFormat::kHWC16:
        layfmt = "kHWC16";
        break;
      case nvinfer1::TensorFormat::kDHWC:
        layfmt = "kDHWC";
        break;
      default:
        break;
    }

    return layfmt;
  };

  printf("\n---\n in-tensor:\n");
  for (int i = 0; i < num_in; ++i) {
    const auto& it = in_desc[i];
    printf(
      "%-32s %-8s %-16s %s\n",
      fmt_dims(it.dims).c_str(),
      fmt_dtype(it.type).c_str(),
      fmt_layout(it.format).c_str(),
      GetTensor(inputs[i], it.dims, it.type, limit).c_str());
  }

  printf(" out-tensor:\n");
  for (int i = 0; i < num_out; ++i) {
    const auto& it = out_desc[i];
    printf("%-32s %-8s %-16s\n", fmt_dims(it.dims).c_str(), fmt_dtype(it.type).c_str(), fmt_layout(it.format).c_str());
  }
}

inline nvinfer1::Dims vec2dims(const std::vector<int>& vec) {
  nvinfer1::Dims dims;
  dims.nbDims = vec.size();
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = vec[i];
  }
  return dims;
}

inline nvinfer1::DataType str2dtype(const std::string& str) {
  nvinfer1::DataType dtype = nvinfer1::DataType::kHALF;
  if (str == "fp16") {
    dtype = nvinfer1::DataType::kHALF;
  } else if (str == "fp32") {
    dtype = nvinfer1::DataType::kFLOAT;
  } else if (str == "int32") {
    dtype = nvinfer1::DataType::kINT32;
  } else if (str == "int64") {
    dtype = nvinfer1::DataType::kINT64;
  } else if (str == "int8") {
    dtype = nvinfer1::DataType::kINT8;
  } else if (str == "fp8") {
    dtype = nvinfer1::DataType::kFP8;
  }
  return dtype;
}

inline nvinfer1::TensorFormat str2layout(const std::string& str) {
  nvinfer1::TensorFormat dtype = nvinfer1::TensorFormat::kLINEAR;
  if (str == "kLINEAR") {
    dtype = nvinfer1::TensorFormat::kLINEAR;
  } else if (str == "kCHW2") {
    dtype = nvinfer1::TensorFormat::kCHW2;
  } else if (str == "kHWC8") {
    dtype = nvinfer1::TensorFormat::kHWC8;
  } else if (str == "kCHW4") {
    dtype = nvinfer1::TensorFormat::kCHW4;
  } else if (str == "kCHW16") {
    dtype = nvinfer1::TensorFormat::kCHW16;
  } else if (str == "kCHW32") {
    dtype = nvinfer1::TensorFormat::kCHW32;
  } else if (str == "kDHWC8") {
    dtype = nvinfer1::TensorFormat::kDHWC8;
  } else if (str == "kCDHW32") {
    dtype = nvinfer1::TensorFormat::kCDHW32;
  } else if (str == "kDLA_LINEAR") {
    dtype = nvinfer1::TensorFormat::kDLA_LINEAR;
  } else if (str == "kDLA_HWC4") {
    dtype = nvinfer1::TensorFormat::kDLA_HWC4;
  } else if (str == "kHWC16") {
    dtype = nvinfer1::TensorFormat::kHWC16;
  } else if (str == "kDHWC") {
    dtype = nvinfer1::TensorFormat::kDHWC;
  }
  return dtype;
}

inline std::string print_dims(const std::vector<int>& vec) {
  std::string str{"["};
  for (int i = 0; i < int(vec.size()); ++i) {
    str.append(std::to_string(vec[i])).append(", ");
  }
  str.append("]");
  return str;
}

// std::string dtype; // col2 (raw string, e.g. "int32")
// std::string layout; // col3

} // namespace utils
