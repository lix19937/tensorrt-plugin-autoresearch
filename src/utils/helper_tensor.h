/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#include <NvInferRuntime.h>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "tensor.h"

namespace trt_edgellm {
namespace rt {
inline std::string getstr(const nvinfer1::Dims& dims) {
  std::string str = "[";
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    str.append(std::to_string(dims.d[i]) + ", ");
  }
  if (str.size() >= 2) {
    str[str.size() - 2] = ']';
    str.resize(str.size() - 1);
  }
  return str;
}

inline nvinfer1::Dims vec2dims(const std::vector<int>& shape) {
  nvinfer1::Dims dims;
  dims.nbDims = shape.size();
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = shape[i];
  }
  return dims;
}

inline std::string getstr(OptionalInputTensors optTensors) {
  std::string str;
  for (auto it : optTensors) {
    str.append(it.get().getShape().formatString());
  }

  return str;
}

inline std::string getstr(OptionalInputTensor optTensor) {
  if (optTensor.has_value()) {
    return optTensor.value().get().getShape().formatString();
  } else {
    return {};
  }
}

inline std::string getstr(OptionalOutputTensor optTensor) {
  if (optTensor.has_value()) {
    return optTensor.value().get().getShape().formatString();
  } else {
    return {};
  }
}

inline std::string getstr(Tensor const& optTensor) {
  return optTensor.getShape().formatString();
}

inline void print(Tensor const& tensor) {
  auto d = tensor.getTRTDims();
  int num = std::accumulate(d.d, &d.d[d.nbDims], 1, std::multiplies<int>{});
  if (num <= 0)
    return;
  if (num > 32)
    num = 32;

  switch (tensor.getDataType()) {
    case nvinfer1::DataType::kFLOAT: {
      std::vector<float> buffer(num);
      CUDA_CHECK(cudaMemcpy(buffer.data(), tensor.rawPointer(), num * sizeof(float), cudaMemcpyDefault));
      for (auto it : buffer) {
        printf("%f, ", it);
      }
      printf("\n");
    } break;

    case nvinfer1::DataType::kHALF: {
      std::vector<half> buffer(num);
      CUDA_CHECK(cudaMemcpy(buffer.data(), tensor.rawPointer(), num * sizeof(half), cudaMemcpyDefault));
      for (auto it : buffer) {
        printf("%f, ", __half2float(it));
      }
      printf("\n");
    } break;

    default:
      break;
  }
}

} // namespace rt
} // namespace trt_edgellm