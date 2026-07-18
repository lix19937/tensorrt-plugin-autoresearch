/**************************************************************
 * @Author: lhm & ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#include <stdio.h>
#include <algorithm>
#include "multiscale_deformable_attention.h"

__device__ void ms_deform_attn_im2col_bilinear_h8c32_fp16(
  const half* bottom_data,
  const int2 sp_hw,
  const int nheads,
  const int channels,
  const float2 hw_im,
  const int m,
  const int c,
  const float weight,
  half2& out12,
  half2& out34) {
  const int h_low = floorf(hw_im.x);
  const int w_low = floorf(hw_im.y);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const float lh = hw_im.x - h_low;
  const float lw = hw_im.y - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = sp_hw.y * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c * 4; // c FROM 0~32 TO 0~7

  half2 factor1_h2 = __float2half2_rn(hh * hw * weight); // 1208us->1162.95us
  half2 factor2_h2 = __float2half2_rn(hh * lw * weight); // 4个factor
  half2 factor3_h2 = __float2half2_rn(lh * hw * weight);
  half2 factor4_h2 = __float2half2_rn(lh * lw * weight);

  const half2* __restrict__ basep = (const half2*)bottom_data;

  if (h_low >= 0 && w_low >= 0) {
    auto ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;

    const half2 input = __ldg(basep + (ptr1 >> 1));
    const half2 input34 = __ldg(basep + ((ptr1 + 2) >> 1));

    out12 = __hfma2(factor1_h2, input, out12);
    out34 = __hfma2(factor1_h2, input34, out34);
  }

  if (h_low >= 0 && w_high <= sp_hw.y - 1) {
    auto ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;

    const half2 input = __ldg(basep + (ptr2 >> 1));
    const half2 input34 = __ldg(basep + ((ptr2 + 2) >> 1));

    out12 = __hfma2(factor2_h2, input, out12);
    out34 = __hfma2(factor2_h2, input34, out34);
  }

  if (h_high <= sp_hw.x - 1 && w_low >= 0) {
    auto ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;

    const half2 input = __ldg(basep + (ptr3 >> 1));
    const half2 input34 = __ldg(basep + ((ptr3 + 2) >> 1));

    out12 = __hfma2(factor3_h2, input, out12);
    out34 = __hfma2(factor3_h2, input34, out34);
  }

  if (h_high <= sp_hw.x - 1 && w_high <= sp_hw.y - 1) {
    auto ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;

    const half2 input = __ldg(basep + (ptr4 >> 1));
    const half2 input34 = __ldg(basep + ((ptr4 + 2) >> 1));

    out12 = __hfma2(factor4_h2, input, out12);
    out34 = __hfma2(factor4_h2, input34, out34);
  }
}

__global__ void ms_deformable_im2col_gpu_kernel_h8c32_c16_fp16(
  const int n,
  const half* __restrict__ data_value,
  const int* __restrict__ data_spatial_shapes,
  const int* __restrict__ data_level_start_index,
  const half* __restrict__ data_sampling_loc,
  const half* __restrict__ data_attn_weight,
  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point,
  half* __restrict__ data_col) {
  int index = (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x) *
      (blockDim.x * blockDim.y * blockDim.z) +
    blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
  if (index > 2 * n - 1)
    return;

  int data_weight_ptr = index / blockDim.x * num_levels * num_point;
  int data_loc_w_ptr = data_weight_ptr << 1;
  int qid_stride = num_heads * channels;
  int data_value_ptr_init_offset = blockIdx.y * spatial_size * qid_stride;
  half2 sum12 = __float2half2_rn(0.f);
  half2 sum34 = __float2half2_rn(0.f);

#pragma unroll
  for (int l_col = 0; l_col < num_levels; ++l_col) {
    int level_start_id = __ldg(data_level_start_index + l_col);
    int spatial_h_ptr = l_col << 1;
    int spatial_h = __ldg(data_spatial_shapes + spatial_h_ptr);
    int spatial_w = __ldg(data_spatial_shapes + spatial_h_ptr + 1);
    int2 spatial_hw{spatial_h, spatial_w};

    const half* data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
#pragma unroll
    for (int p_col = 0; p_col < num_point; ++p_col) {
      half loc_w = __ldg(data_sampling_loc + data_loc_w_ptr);
      half loc_h = __ldg(data_sampling_loc + data_loc_w_ptr + 1);
      float2 loc_hw{loc_h, loc_w};
      // or use follow -------------------------------------
      // half2 loc_wh = __ldg((const half2*)data_sampling_loc + data_loc_w_ptr / 2);
      // float2 loc_hw{loc_wh.x, loc_wh.y};
      // ----------------------------------------------------

      half weight = __ldg(data_attn_weight + data_weight_ptr);

      float2 hw_im;
      hw_im.x = loc_hw.x * spatial_hw.x - 0.5f;
      hw_im.y = loc_hw.y * spatial_hw.y - 0.5f;

      if (hw_im.x > -1 && hw_im.y > -1 && hw_im.x < spatial_hw.x && hw_im.y < spatial_hw.y) {
        ms_deform_attn_im2col_bilinear_h8c32_fp16(
          data_value_ptr, spatial_hw, num_heads, channels, hw_im, threadIdx.y, threadIdx.x, weight, sum12, sum34);
      }
      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
    }
  }
  half2* data_col_ptr = (half2*)data_col + (index << 1);
  *data_col_ptr = sum12;

  half2* data_col_ptr1 = (half2*)data_col + (index << 1) + 1;
  *data_col_ptr1 = sum34;
}

__global__ void ms_deformable_im2col_gpu_kernel_fp16(
  const int n,
  const half* __restrict__ data_value,
  const int* __restrict__ data_spatial_shapes,
  const int* __restrict__ data_level_start_index,
  const half* __restrict__ data_sampling_loc,
  const half* __restrict__ data_attn_weight,
  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point,
  half* __restrict__ data_col) {
  int index =
    (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  if (index > 2 * n - 1)
    return;

  const int c_col = (index * 4) % channels;
  const int m_col = (index * 4 / channels) % num_heads;

  int data_weight_ptr = (index * 4 / channels) * num_levels * num_point;
  int data_loc_w_ptr = data_weight_ptr << 1;
  int qid_stride = num_heads * channels;
  int data_value_ptr_init_offset = blockIdx.y * spatial_size * qid_stride;
  half2 sum12 = __float2half2_rn(0.f);
  half2 sum34 = __float2half2_rn(0.f);

//#pragma unroll
  for (int l_col = 0; l_col < num_levels; ++l_col) {
    int level_start_id = __ldg(data_level_start_index + l_col);
    int spatial_h_ptr = l_col << 1;
    int spatial_h = __ldg(data_spatial_shapes + spatial_h_ptr);
    int spatial_w = __ldg(data_spatial_shapes + spatial_h_ptr + 1);
    int2 spatial_hw{spatial_h, spatial_w};

    const half* data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
//#pragma unroll
    for (int p_col = 0; p_col < num_point; ++p_col) {
      half loc_w = __ldg(data_sampling_loc + data_loc_w_ptr);
      half loc_h = __ldg(data_sampling_loc + data_loc_w_ptr + 1);
      float2 loc_hw{loc_h, loc_w};
      // or use follow 
      // half2 loc_wh = ((half2*)data_sampling_loc)[data_loc_w_ptr / 2];
      // float2 loc_hw{loc_wh.x, loc_wh.y};

      half weight = __ldg(data_attn_weight + data_weight_ptr);

      float2 hw_im;
      hw_im.x = loc_hw.x * spatial_hw.x - 0.5f;
      hw_im.y = loc_hw.y * spatial_hw.y - 0.5f;

      if (hw_im.x > -1 && hw_im.y > -1 && hw_im.x < spatial_hw.x && hw_im.y < spatial_hw.y) {
        ms_deform_attn_im2col_bilinear_h8c32_fp16(
          data_value_ptr, spatial_hw, num_heads, channels, hw_im, m_col, c_col / 4, weight, sum12, sum34);
      }
      data_weight_ptr += 1;
      data_loc_w_ptr += 2;
    }
  }
  half2* data_col_ptr = &((half2*)data_col)[index << 1];
  *data_col_ptr = sum12;
  half2* data_col_ptr1 = &((half2*)data_col)[(index << 1) + 1];
  *data_col_ptr1 = sum34;
}

int ms_deformable_im2col_cuda(
  cudaStream_t stream,
  const half* value,
  const int* spatial_shapes,
  const int* level_start_index,
  const half* sampling_locations,
  const half* attention_weights,
  half* output,

  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point) {
  auto im2col_step_ = batch_size; // std::min(batch_size, im2col_step);

  int batch_n = im2col_step_; //
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto per_sample_loc_size = per_attn_weight_size * 2;

  for (int n = 0; n < batch_size / im2col_step_; ++n) {
    auto data_col = output + n * batch_n * num_query * num_heads * channels;
    auto data_value = value + n * im2col_step_ * per_value_size;
    auto data_sampling_loc = sampling_locations + n * im2col_step_ * per_sample_loc_size;
    auto data_attn_weight = attention_weights + n * im2col_step_ * per_attn_weight_size;
    int num_actual_kernels = num_query * num_heads * channels;

    int num_threads = 1024;
    if (num_heads == 8 && (channels == 32 || channels == 16)) {
      const int block_x = channels / 4; //  half2 out1 , half2 out2
      const int block_y = num_heads;
      const int block_z = num_threads / 4 / block_y / block_x; // 4 or 8

      dim3 block(block_x, block_y, block_z);
      dim3 grid((num_actual_kernels + num_threads - 1) / num_threads, batch_n);
      ms_deformable_im2col_gpu_kernel_h8c32_c16_fp16<<<grid, block, 0, stream>>>(
        num_actual_kernels,
        data_value,
        spatial_shapes,
        level_start_index,
        data_sampling_loc,
        data_attn_weight,
        batch_n,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        data_col);

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
      }
    } else {
      dim3 block(num_threads / 4, 1);
      dim3 grid((num_actual_kernels + num_threads - 1) / num_threads, batch_n);
      ms_deformable_im2col_gpu_kernel_fp16<<<grid, block, 0, stream>>>(
        num_actual_kernels,
        data_value,
        spatial_shapes,
        level_start_index,
        data_sampling_loc,
        data_attn_weight,
        batch_n,
        spatial_size,
        num_heads,
        channels,
        num_levels,
        num_query,
        num_point,
        data_col);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
      }
    }
  }
  return 0;
}
