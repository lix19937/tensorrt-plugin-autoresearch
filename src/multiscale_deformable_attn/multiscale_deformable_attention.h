/**************************************************************
 * @Author: lhm & ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

int ms_deformable_im2col_cuda(
  cudaStream_t stream,
  const float* data_value,
  const int* data_spatial_shapes,
  const int* data_level_start_index,
  const float* data_sampling_locations,
  const float* data_attention_weights,
  float* output,
  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point);

int ms_deformable_im2col_cuda(
  cudaStream_t stream,
  const half* data_value,
  const int* data_spatial_shapes,
  const int* data_level_start_index,
  const half* data_sampling_locations,
  const half* data_attention_weights,
  half* output,
  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point);
