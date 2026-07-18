

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdio>

#include <algorithm>
#include <cmath>
#include <vector>

#include <unistd.h>
#include <cstdio>
#include <cuda/std/limits>
#include "fused_sca_deform_attn.h"
#include "helper_cuda.h"
// #include "helper.h"

template <typename T>
__forceinline__ __device__ T hmax(const T& a, const T& b) {
  return max(a, b);
}

#if __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ __half hmax(const __half& a, const __half& b) {
  return __hmax(a, b);
}
template <>
__forceinline__ __device__ __half2 hmax(const __half2& a, const __half2& b) {
  return __hmax2(a, b);
}
#else
template <>
__forceinline__ __device__ __half hmax(const __half& a, const __half& b) {
  return __hgt(a, b) ? a : b;
}
template <>
__forceinline__ __device__ __half2 hmax(const __half2& a, const __half2& b) {
  return __hfma2(__hgt2(a, b), a, __hmul2(__hle2(a, b), b));
}
#endif

// ---- float, 32 channels: 8 float4 ----
__device__ __forceinline__ void sca_bilinear_gather32_f32(
  const float* __restrict__ bottom_data,
  const int2 sp_hw,
  const int nheads,
  const int channels,
  const float2 hw_im,
  const int m,
  const int c,
  const float weight,
  float4* out) { // out[8]
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
  const int base_ptr = m * channels + c * 32;
  const float4* __restrict__ vp = (const float4*)bottom_data;
  const float f1 = hh * hw * weight, f2 = hh * lw * weight;
  const float f3 = lh * hw * weight, f4 = lh * lw * weight;
  auto acc = [&](const int p, const float f) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 a = __ldg(vp + p + i);
      out[i].x += f * a.x;
      out[i].y += f * a.y;
      out[i].z += f * a.z;
      out[i].w += f * a.w;
    }
  };
  if (h_low >= 0 && w_low >= 0)
    acc((h_low_ptr_offset + w_low_ptr_offset + base_ptr) >> 2, f1);
  if (h_low >= 0 && w_high <= sp_hw.y - 1)
    acc((h_low_ptr_offset + w_high_ptr_offset + base_ptr) >> 2, f2);
  if (h_high <= sp_hw.x - 1 && w_low >= 0)
    acc((h_high_ptr_offset + w_low_ptr_offset + base_ptr) >> 2, f3);
  if (h_high <= sp_hw.x - 1 && w_high <= sp_hw.y - 1)
    acc((h_high_ptr_offset + w_high_ptr_offset + base_ptr) >> 2, f4);
}

// ---- half, 32 channels: 16 half2 ----
__device__ __forceinline__ void sca_bilinear_gather32_fp16(
  const __half* __restrict__ bottom_data,
  const int2 sp_hw,
  const int nheads,
  const int channels,
  const float2 hw_im,
  const int m,
  const int c,
  const float weight,
  __half2* out) { // out[16]
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
  const int base_ptr = m * channels + c * 32;
  const float4* __restrict__ vp = (const float4*)bottom_data;
  const __half2 f1 = __float2half2_rn(hh * hw * weight);
  const __half2 f2 = __float2half2_rn(hh * lw * weight);
  const __half2 f3 = __float2half2_rn(lh * hw * weight);
  const __half2 f4 = __float2half2_rn(lh * lw * weight);
  auto acc = [&](const int p, const __half2 f) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 raw = __ldg(vp + p + i);
      const __half2* rh = (const __half2*)&raw;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out[i * 4 + j] = __hfma2(f, rh[j], out[i * 4 + j]);
    }
  };
  if (h_low >= 0 && w_low >= 0)
    acc((h_low_ptr_offset + w_low_ptr_offset + base_ptr) >> 3, f1);
  if (h_low >= 0 && w_high <= sp_hw.y - 1)
    acc((h_low_ptr_offset + w_high_ptr_offset + base_ptr) >> 3, f2);
  if (h_high <= sp_hw.x - 1 && w_low >= 0)
    acc((h_high_ptr_offset + w_low_ptr_offset + base_ptr) >> 3, f3);
  if (h_high <= sp_hw.x - 1 && w_high <= sp_hw.y - 1)
    acc((h_high_ptr_offset + w_high_ptr_offset + base_ptr) >> 3, f4);
}

// ============================ 32 channels / thread ===========================

__global__ __launch_bounds__(64, 12) void sca_deform_attn_kernel_f32(
  const int num_query,
  const int num_point,
  const int points_per_group,
  const float* __restrict__ data_value,
  const int32_t* __restrict__ data_spatial_shapes,
  const float* __restrict__ data_reference_points,
  const float* __restrict__ data_sampling_offsets,
  const float* __restrict__ data_attn_weight,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int out_cols, // 840
  const int grid_w, // 184
  float* __restrict__ data_col) {
  const int batch_index = blockIdx.y;
  const int query_index = blockIdx.x * blockDim.z + threadIdx.z;
  if (query_index >= num_query)
    return;
  const int head_index = threadIdx.y;
  const int c_group = threadIdx.x; // 32 channels per thread

  const int qid_stride = num_heads * channels;
  const float* data_value_ptr = data_value + batch_index * spatial_size * qid_stride;
  int data_weight_ptr = ((batch_index * num_query + query_index) * num_heads + head_index) * num_levels * num_point;
  int data_offset_w_ptr = data_weight_ptr << 1;
  const int data_points_ptr = (batch_index * num_query + query_index) * points_per_group * 2;

  const int all_points = num_levels * num_point;
  float max_weight = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
  for (int w_index = 0; w_index < all_points; ++w_index) {
    max_weight = fmaxf(max_weight, __ldg(&data_attn_weight[data_weight_ptr + w_index]));
  }

  float4 out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    out[i] = {0.f, 0.f, 0.f, 0.f};
  float sum_weight = 0.f;
  for (int level_index = 0; level_index < num_levels; ++level_index) {
    const int spatial_h_ptr = level_index << 1;
    const int spatial_h = __ldg(&data_spatial_shapes[spatial_h_ptr]);
    const int spatial_w = __ldg(&data_spatial_shapes[spatial_h_ptr + 1]);
    const int2 spatial_hw{spatial_h, spatial_w};

    for (int point_index = 0; point_index < num_point; ++point_index) {
      const int point_index_per_group = point_index % points_per_group;
      const float reference_point_x = __ldg(&data_reference_points[data_points_ptr + point_index_per_group * 2]);
      const float reference_point_y = __ldg(&data_reference_points[data_points_ptr + point_index_per_group * 2 + 1]);
      const float offset_x = __ldg(&data_sampling_offsets[data_offset_w_ptr]);
      const float offset_y = __ldg(&data_sampling_offsets[data_offset_w_ptr + 1]);
      const float loc_w = reference_point_x * spatial_w + offset_x;
      const float loc_h = reference_point_y * spatial_h + offset_y;

      const float weight = expf(__ldg(&data_attn_weight[data_weight_ptr]) - max_weight);
      sum_weight += weight;

      float2 hw_im;
      hw_im.x = loc_h - 0.5f;
      hw_im.y = loc_w - 0.5f;
      if (hw_im.x > -1 && hw_im.y > -1 && hw_im.x < spatial_h && hw_im.y < spatial_w) {
        sca_bilinear_gather32_f32(
          data_value_ptr, spatial_hw, num_heads, channels, hw_im, head_index, c_group, weight, out);
      }
      data_weight_ptr += 1;
      data_offset_w_ptr += 2;
    }
    data_value_ptr += spatial_h * spatial_w * qid_stride;
  }
  const float inv_sum = 1.f / sum_weight;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i].x *= inv_sum;
    out[i].y *= inv_sum;
    out[i].z *= inv_sum;
    out[i].w *= inv_sum;
  }
  // Fused post-plugin permute + reshape: R[0, cc, b*grid_h + q//grid_w, q%grid_w] = O[b, q, cc].
  // 32 channels (cc..cc+31) strided by out_cols*grid_w in R.
  const int cc = head_index * channels + c_group * 32;
  const int grid_h = out_cols / gridDim.y;
  const int qh = query_index / grid_w;
  const int qw = query_index - qh * grid_w;
  const int row = batch_index * grid_h + qh;
  const int stride = out_cols * grid_w;
  const int base = (cc * out_cols + row) * grid_w + qw;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __stcs(&data_col[base + (i * 4 + 0) * stride], out[i].x);
    __stcs(&data_col[base + (i * 4 + 1) * stride], out[i].y);
    __stcs(&data_col[base + (i * 4 + 2) * stride], out[i].z);
    __stcs(&data_col[base + (i * 4 + 3) * stride], out[i].w);
  }
}

__global__ __launch_bounds__(1024) void sca_deform_attn_kernel_fp16(
  const int num_query,
  const int num_point,
  const int points_per_group,
  const __half* __restrict__ data_value,
  const int32_t* __restrict__ data_spatial_shapes,
  const __half* __restrict__ data_reference_points,
  const __half* __restrict__ data_sampling_offsets,
  const __half* __restrict__ data_attn_weight,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int out_cols, // 840
  const int grid_w, // 184
  __half* __restrict__ data_col) {
  const int batch_index = blockIdx.y;
  const int query_index = blockIdx.x * blockDim.z + threadIdx.z;
  if (query_index >= num_query)
    return;
  const int head_index = threadIdx.y;
  const int c_group = threadIdx.x;

  const int qid_stride = num_heads * channels;
  const __half* data_value_ptr = data_value + batch_index * spatial_size * qid_stride;
  int data_weight_ptr = ((batch_index * num_query + query_index) * num_heads + head_index) * num_levels * num_point;
  int data_offset_w_ptr = data_weight_ptr << 1;
  const int data_points_ptr = (batch_index * num_query + query_index) * points_per_group * 2;

  const int all_points = num_levels * num_point;
  __half max_weight = -cuda::std::numeric_limits<__half>::infinity();
#pragma unroll
  for (int w_index = 0; w_index < all_points; ++w_index) {
    max_weight = hmax(max_weight, __ldg(&data_attn_weight[data_weight_ptr + w_index]));
  }

  __half2 out[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    out[i] = __float2half2_rn(0.f);
  __half sum_weight = __float2half(0.f);
  for (int level_index = 0; level_index < num_levels; ++level_index) {
    const int spatial_h_ptr = level_index << 1;
    const int spatial_h = __ldg(&data_spatial_shapes[spatial_h_ptr]);
    const int spatial_w = __ldg(&data_spatial_shapes[spatial_h_ptr + 1]);
    const int2 spatial_hw{spatial_h, spatial_w};

    for (int point_index = 0; point_index < num_point; ++point_index) {
      const int point_index_per_group = point_index % points_per_group;
      const __half reference_point_x = __ldg(&data_reference_points[data_points_ptr + point_index_per_group * 2]);
      const __half reference_point_y = __ldg(&data_reference_points[data_points_ptr + point_index_per_group * 2 + 1]);
      const __half offset_x = __ldg(&data_sampling_offsets[data_offset_w_ptr]);
      const __half offset_y = __ldg(&data_sampling_offsets[data_offset_w_ptr + 1]);
      const float loc_w = __half2float(reference_point_x) * spatial_w + __half2float(offset_x);
      const float loc_h = __half2float(reference_point_y) * spatial_h + __half2float(offset_y);

      const __half weight = hexp(__hsub(__ldg(&data_attn_weight[data_weight_ptr]), max_weight));
      sum_weight = __hadd(sum_weight, weight);

      float2 hw_im;
      hw_im.x = loc_h - 0.5f;
      hw_im.y = loc_w - 0.5f;
      if (hw_im.x > -1 && hw_im.y > -1 && hw_im.x < spatial_h && hw_im.y < spatial_w) {
        sca_bilinear_gather32_fp16(
          data_value_ptr, spatial_hw, num_heads, channels, hw_im, head_index, c_group, __half2float(weight), out);
      }
      data_weight_ptr += 1;
      data_offset_w_ptr += 2;
    }
    data_value_ptr += spatial_h * spatial_w * qid_stride;
  }
  const __half2 inv_sum = __half2half2(__hdiv(__float2half(1.f), sum_weight));
#pragma unroll
  for (int i = 0; i < 16; ++i)
    out[i] = __hmul2(out[i], inv_sum);
  // Fused post-plugin permute + reshape: R[0, cc, b*grid_h + q//grid_w, q%grid_w] = O[b, q, cc].
  // 32 channels (cc..cc+31) strided by out_cols*grid_w in R.
  const int cc = head_index * channels + c_group * 32;
  const int grid_h = out_cols / gridDim.y;
  const int qh = query_index / grid_w;
  const int qw = query_index - qh * grid_w;
  const int row = batch_index * grid_h + qh;
  const int stride = out_cols * grid_w;
  const int base = (cc * out_cols + row) * grid_w + qw;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const __half* oh = (const __half*)&out[i];
    __stcs(&data_col[base + (i * 2 + 0) * stride], oh[0]);
    __stcs(&data_col[base + (i * 2 + 1) * stride], oh[1]);
  }
}

// block_z = 每 block 覆盖的 query 数; 1024/(block_x*block_y) 但不超过 cap
// (宽核寄存器多, cap 收紧防 spill), 下限 1 保证每个 query 都被覆盖。
namespace {
inline int pick_block_z(int block_x, int block_y, int cap = 8) {
  int bz = 1024 / (block_x * block_y);
  if (bz < 1)
    bz = 1;
  if (bz > cap)
    bz = cap;
  return bz;
}
} // namespace

void sca_deform_attn_f32(
  const float* data_value,
  const int32_t* data_spatial_shapes,
  const float* data_reference_points,
  const float* data_sampling_offsets,
  const float* data_attn_weight,
  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point,
  const int points_per_group,
  const int out_cols,
  const int grid_w,
  float* data_col,
  cudaStream_t stream) {
  const int block_x = channels / 32; // 32 channels per thread
  const int block_y = num_heads;
  const int block_z = pick_block_z(block_x, block_y, 8); // cap 8: more queries/block hides load latency
  dim3 block(block_x, block_y, block_z);
  dim3 grid((num_query + block_z - 1) / block_z, batch_size);
  sca_deform_attn_kernel_f32<<<grid, block, 0, stream>>>(
    num_query,
    num_point,
    points_per_group,
    data_value,
    data_spatial_shapes,
    data_reference_points,
    data_sampling_offsets,
    data_attn_weight,
    spatial_size,
    num_heads,
    channels,
    num_levels,
    out_cols,
    grid_w,
    data_col);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in sca_deform_attn_f32: %s\n", cudaGetErrorString(err));
  }
}

void sca_deform_attn_fp16(
  const __half* data_value,
  const int32_t* data_spatial_shapes,
  const __half* data_reference_points,
  const __half* data_sampling_offsets,
  const __half* data_attn_weight,
  const int batch_size,
  const int spatial_size,
  const int num_heads,
  const int channels,
  const int num_levels,
  const int num_query,
  const int num_point,
  const int points_per_group,
  const int out_cols,
  const int grid_w,
  __half* data_col,
  cudaStream_t stream) {
  const int block_x = channels / 32;
  const int block_y = num_heads;
  const int block_z = pick_block_z(block_x, block_y, 16); // cap 16: lighter regs after 128b loads
  dim3 block(block_x, block_y, block_z);
  dim3 grid((num_query + block_z - 1) / block_z, batch_size);
  sca_deform_attn_kernel_fp16<<<grid, block, 0, stream>>>(
    num_query,
    num_point,
    points_per_group,
    data_value,
    data_spatial_shapes,
    data_reference_points,
    data_sampling_offsets,
    data_attn_weight,
    spatial_size,
    num_heads,
    channels,
    num_levels,
    out_cols,
    grid_w,
    data_col);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in sca_deform_attn_fp16: %s\n", cudaGetErrorString(err));
  }
}

// =============================================================================
// Fused restore-bev grid_sample(nearest, align_corners=True) + camera reduce_sum
// (替代权重全 1 的 query_reduce_sum grouped-conv) + permute + mul_pillarweight。
//
// 一次吃掉 attention.py 里 restore_outputs 的整段:
//   bev_queries = F.grid_sample(queries_out, restore_bev_grid,
//                               mode="nearest", align_corners=True)   # [1,256,840,184]
//   bev_queries = bev_queries.reshape(bs, -1, bev_h, bev_w)            # [1,1792,120,184]
//   slots = query_reduce_sum(bev_queries).flatten(-2).permute(0,2,1)   # [1,22080,256]
//   slots = mul_pillarweight.mul(slots, counts)                        # [1,22080,256]
//
// 输入:
//   R      = queries_out = [1, embed_dims, H_r, W_r] = [1, 256, 840, 184]
//            H_r = num_cams*bev_h (=7*120=840), W_r = bev_w (=184), NCHW (W 最快)。
//   grid   = restore_bev_grid = [1, H_r, W_r, 2]  (gx=W 归一化 [-1,1], gy=H)。
//   counts = [1, Q, 1] = [1, 22080, 1], Q = bev_h*bev_w = 120*184。
// 输出:
//   slots  = [1, Q, embed_dims] = [1, 22080, 256]  (q 外, c 内)。
//
// 数学等价 (逐元素对齐 PyTorch 参考):
//   query_reduce_sum(权重全1, groups=embed_dims) 对 [1,C*num_cams,bev_h,bev_w]
//     == bev.reshape(1,C,num_cams,bev_h*bev_w).sum(2)
//     == Σ_{scam} grid_sample_out[0, c, ho=scam*bev_h+bh, wo=bw]
//   grid_sample nearest(AC=True) + reshape:
//     src_w = round((gx+1)/2*(W_r-1)), src_h = round((gy+1)/2*(H_r-1))
//     越界 -> 0 (zeros padding, PyTorch 默认, 与 gridsample_reduce 一致)
//   合并:
//     q  = bh*bev_w + bw
//     slots[0,q,c] = ( Σ_{scam} in_bounds? R[0,c,src_h,src_w] : 0 ) * counts[0,q,0]
//
// 索引表: grid (相机->BEV 投影, 来自标定) 在多次推理间恒定, 故每个
//   (scam,bh,bw) 的源空间偏移 spatial = src_h*W_r + src_w 是固定的, 越界记 -1。
//   表布局 table[(scam*bev_h + bh)*bev_w + bw], 大小 num_cams*bev_h*bev_w ints。
//   c 维在 gather 时再加 c*H_r*W_r。
// =============================================================================

// ---- 构建常量索引表 (只依赖 grid) ----
// grid 元素类型随 value (float 或 __half); 用 __half2 一次读或 float 两次读。
__device__ __forceinline__ void read_grid_xy(const float* __restrict__ grid, size_t p, float& gx, float& gy) {
  gx = grid[p * 2 + 0];
  gy = grid[p * 2 + 1];
}
__device__ __forceinline__ void read_grid_xy(const __half* __restrict__ grid, size_t p, float& gx, float& gy) {
  const __half2 g = ((const __half2*)grid)[p];
  gx = __low2float(g);
  gy = __high2float(g);
}

template <typename GT, int AC>
__global__ void fused_reduce_build_index_cuda_kernel(
  const GT* __restrict__ grid, // [1, H_r, W_r, 2]
  int* __restrict__ table, // [num_cams, bev_h, bev_w]
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w) {
  const long n = (long)num_cams * bev_h * bev_w;
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  const int bw = idx % bev_w;
  const int bh = (idx / bev_w) % bev_h;
  const int scam = idx / (bev_w * bev_h);
  // (scam, bh, bw) -> grid 空间坐标 (ho, wo)
  const int ho = scam * bev_h + bh;
  const int wo = bw;
  float gx, gy;
  read_grid_xy(grid, (size_t)ho * W_r + wo, gx, gy);
  int src_w, src_h;
  if (AC) {
    src_w = __float2int_rn((gx + 1.f) * 0.5f * (W_r - 1));
    src_h = __float2int_rn((gy + 1.f) * 0.5f * (H_r - 1));
  } else {
    src_w = (int)floorf((gx + 1.f) * W_r * 0.5f);
    src_h = (int)floorf((gy + 1.f) * H_r * 0.5f);
  }
  int base = -1; // 越界 -> zeros padding
  if (src_w >= 0 && src_w < W_r && src_h >= 0 && src_h < H_r) {
    base = src_h * W_r + src_w; // 空间偏移 (c 维后加)
  }
  table[idx] = base;
}

// ---- float: gather + reduce_sum + mul ----
// 每线程负责 QPT 个相邻 query 的全部 C 通道: 复用 c*HW 地址计算, 用 ILP 隐藏
// 分散 gather 的访存延迟。生产环境 grid 平滑时相邻 query 的 base 近似连续。
template <int QPT>
__global__ void fused_reduce_gather_mul_cuda_kernel_f(
  const float* __restrict__ R, // [1, embed_dims, H_r, W_r]
  const int* __restrict__ table, // [num_cams, bev_h, bev_w]
  const float* __restrict__ counts, // [1, Q, 1]
  float* __restrict__ slots, // [1, Q, embed_dims]
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w,
  int C) {
  const int bw0 = (blockIdx.x * blockDim.x + threadIdx.x) * QPT; // 首 query 列
  const int bh = blockIdx.y;
  if (bw0 >= bev_w)
    return;
  const long HW = (long)H_r * W_r;

  int base[QPT][8];
  float w[QPT];
  bool act[QPT];
#pragma unroll
  for (int j = 0; j < QPT; ++j) {
    const int bw = bw0 + j;
    act[j] = bw < bev_w;
#pragma unroll
    for (int scam = 0; scam < num_cams; ++scam) {
      const long row = ((long)scam * bev_h + bh) * bev_w;
      base[j][scam] = act[j] ? table[row + bw] : -1;
    }
    w[j] = act[j] ? counts[bh * bev_w + bw] : 0.f;
  }
  const int q0 = bh * bev_w + bw0;

  for (int c = 0; c < C; ++c) {
    const float* __restrict__ Rc = R + (long)c * HW;
    float acc[QPT];
#pragma unroll
    for (int j = 0; j < QPT; ++j)
      acc[j] = 0.f;
#pragma unroll
    for (int scam = 0; scam < num_cams; ++scam) {
#pragma unroll
      for (int j = 0; j < QPT; ++j) {
        const int b = base[j][scam];
        if (b >= 0)
          acc[j] += __ldg(&Rc[b]);
      }
    }
#pragma unroll
    for (int j = 0; j < QPT; ++j)
      if (act[j])
        slots[(long)(q0 + j) * C + c] = acc[j] * w[j];
  }
}

// ---- half: gather + reduce_sum + mul ----
// 与 fp32 同构: 每线程 QPT 个相邻 query, 复用 c*HW + ILP 隐藏 gather 延迟。
// 累加用 float 保精度 (与参考对齐), 输出转 half。
template <int QPT>
__global__ void fused_reduce_gather_mul_cuda_kernel_h(
  const __half* __restrict__ R,
  const int* __restrict__ table,
  const __half* __restrict__ counts,
  __half* __restrict__ slots,
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w,
  int C) {
  const int bw0 = (blockIdx.x * blockDim.x + threadIdx.x) * QPT;
  const int bh = blockIdx.y;
  if (bw0 >= bev_w)
    return;
  const long HW = (long)H_r * W_r;

  int base[QPT][8];
  float w[QPT];
  bool act[QPT];
#pragma unroll
  for (int j = 0; j < QPT; ++j) {
    const int bw = bw0 + j;
    act[j] = bw < bev_w;
#pragma unroll
    for (int scam = 0; scam < num_cams; ++scam) {
      const long row = ((long)scam * bev_h + bh) * bev_w;
      base[j][scam] = act[j] ? table[row + bw] : -1;
    }
    w[j] = act[j] ? __half2float(counts[bh * bev_w + bw]) : 0.f;
  }
  const int q0 = bh * bev_w + bw0;

  for (int c = 0; c < C; ++c) {
    const __half* __restrict__ Rc = R + (long)c * HW;
    float acc[QPT];
#pragma unroll
    for (int j = 0; j < QPT; ++j)
      acc[j] = 0.f;
#pragma unroll
    for (int scam = 0; scam < num_cams; ++scam) {
#pragma unroll
      for (int j = 0; j < QPT; ++j) {
        const int b = base[j][scam];
        if (b >= 0)
          acc[j] += __half2float(__ldg(&Rc[b]));
      }
    }
#pragma unroll
    for (int j = 0; j < QPT; ++j)
      if (act[j])
        slots[(long)(q0 + j) * C + c] = __float2half(acc[j] * w[j]);
  }
}

// // ---- int8: gather + reduce_sum + mul ----
// // R 存 int8 (对称量化, 真实值 = q*scaleR)。每线程 QPT 个 query, 累加用 int32
// // (7 个相机 * 127 << 2^31, 无溢出), 最后 * scaleR * counts 一次转 half。
// // 生产平滑 grid 下相邻 query 读相邻 int8 -> 一个 32B sector 服务 32 个 query, 读带宽 4x。
// template <int QPT>
// __global__ void fused_reduce_gather_mul_kernel_i8(
//     const int8_t* __restrict__ R,
//     float scaleR,
//     const int* __restrict__ table,
//     const __half* __restrict__ counts,
//     __half* __restrict__ slots,
//     int num_cams,
//     int H_r,
//     int W_r,
//     int bev_h,
//     int bev_w,
//     int C) {
//   const int bw0 = (blockIdx.x * blockDim.x + threadIdx.x) * QPT;
//   const int bh = blockIdx.y;
//   if (bw0 >= bev_w)
//     return;
//   const long HW = (long)H_r * W_r;

//   int base[QPT][8];
//   float w[QPT];
//   bool act[QPT];
// #pragma unroll
//   for (int j = 0; j < QPT; ++j) {
//     const int bw = bw0 + j;
//     act[j] = bw < bev_w;
// #pragma unroll
//     for (int scam = 0; scam < num_cams; ++scam) {
//       const long row = ((long)scam * bev_h + bh) * bev_w;
//       base[j][scam] = act[j] ? table[row + bw] : -1;
//     }
//     w[j] = act[j] ? __half2float(counts[bh * bev_w + bw]) : 0.f;
//   }
//   const int q0 = bh * bev_w + bw0;

//   for (int c = 0; c < C; ++c) {
//     const int8_t* __restrict__ Rc = R + (long)c * HW;
//     int acc[QPT];
// #pragma unroll
//     for (int j = 0; j < QPT; ++j)
//       acc[j] = 0;
// #pragma unroll
//     for (int scam = 0; scam < num_cams; ++scam) {
// #pragma unroll
//       for (int j = 0; j < QPT; ++j) {
//         const int b = base[j][scam];
//         if (b >= 0)
//           acc[j] += (int)__ldg(&Rc[b]);
//       }
//     }
// #pragma unroll
//     for (int j = 0; j < QPT; ++j)
//       if (act[j])
//         slots[(long)(q0 + j) * C + c] = __float2half((float)acc[j] * scaleR * w[j]);
//   }
// }

// // ---- fp8 (e4m3): gather + reduce_sum + mul ----
// // R 存 fp8_e4m3 (无需 scale, 直接编码近似值)。累加用 float。
// template <int QPT>
// __global__ void fused_reduce_gather_mul_kernel_fp8(
//     const __nv_fp8_e4m3* __restrict__ R,
//     const int* __restrict__ table,
//     const __half* __restrict__ counts,
//     __half* __restrict__ slots,
//     int num_cams,
//     int H_r,
//     int W_r,
//     int bev_h,
//     int bev_w,
//     int C) {
//   const int bw0 = (blockIdx.x * blockDim.x + threadIdx.x) * QPT;
//   const int bh = blockIdx.y;
//   if (bw0 >= bev_w)
//     return;
//   const long HW = (long)H_r * W_r;

//   int base[QPT][8];
//   float w[QPT];
//   bool act[QPT];
// #pragma unroll
//   for (int j = 0; j < QPT; ++j) {
//     const int bw = bw0 + j;
//     act[j] = bw < bev_w;
// #pragma unroll
//     for (int scam = 0; scam < num_cams; ++scam) {
//       const long row = ((long)scam * bev_h + bh) * bev_w;
//       base[j][scam] = act[j] ? table[row + bw] : -1;
//     }
//     w[j] = act[j] ? __half2float(counts[bh * bev_w + bw]) : 0.f;
//   }
//   const int q0 = bh * bev_w + bw0;

//   for (int c = 0; c < C; ++c) {
//     const __nv_fp8_e4m3* __restrict__ Rc = R + (long)c * HW;
//     float acc[QPT];
// #pragma unroll
//     for (int j = 0; j < QPT; ++j)
//       acc[j] = 0.f;
// #pragma unroll
//     for (int scam = 0; scam < num_cams; ++scam) {
// #pragma unroll
//       for (int j = 0; j < QPT; ++j) {
//         const int b = base[j][scam];
//         if (b >= 0)
//           acc[j] += (float)Rc[b];
//       }
//     }
// #pragma unroll
//     for (int j = 0; j < QPT; ++j)
//       if (act[j])
//         slots[(long)(q0 + j) * C + c] = __float2half(acc[j] * w[j]);
//   }
// }

// ---- host wrappers ----
void fused_reduce_build_index_f(
  const float* grid,
  int* table,
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w,
  int align_corners,
  cudaStream_t stream) {
  const long n = (long)num_cams * bev_h * bev_w;
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  if (align_corners)
    fused_reduce_build_index_cuda_kernel<float, 1>
      <<<blocks, threads, 0, stream>>>(grid, table, num_cams, H_r, W_r, bev_h, bev_w);
  else
    fused_reduce_build_index_cuda_kernel<float, 0>
      <<<blocks, threads, 0, stream>>>(grid, table, num_cams, H_r, W_r, bev_h, bev_w);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in fused_reduce_build_index_cuda_f: %s\n", cudaGetErrorString(err));
  }
}

void fused_reduce_build_index_h(
  const __half* grid,
  int* table,
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w,
  int align_corners,
  cudaStream_t stream) {
  const long n = (long)num_cams * bev_h * bev_w;
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  if (align_corners)
    fused_reduce_build_index_cuda_kernel<__half, 1>
      <<<blocks, threads, 0, stream>>>(grid, table, num_cams, H_r, W_r, bev_h, bev_w);
  else
    fused_reduce_build_index_cuda_kernel<__half, 0>
      <<<blocks, threads, 0, stream>>>(grid, table, num_cams, H_r, W_r, bev_h, bev_w);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in fused_reduce_build_index_cuda_h: %s\n", cudaGetErrorString(err));
  }
}

// R: [1,256,840,184], grid: [1,840,184,2], counts: [1,22080,1] -> slots [1,22080,256]
void fused_gridsample_reduce_mul_f(
  const float* R,
  const float* counts,
  float* slots,
  const int* table,
  int embed_dims,
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w,
  cudaStream_t stream) {
  const int TX = 128;
  const int QPT = 2; // 每线程 query 数 (2 与 4 实测持平, 取 2 省寄存器)
  const int cols = (bev_w + QPT - 1) / QPT;
  dim3 grid_dim((cols + TX - 1) / TX, bev_h);
  fused_reduce_gather_mul_cuda_kernel_f<2>
    <<<grid_dim, TX, 0, stream>>>(R, table, counts, slots, num_cams, H_r, W_r, bev_h, bev_w, embed_dims);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in fused_gridsample_reduce_mul_cuda_f: %s\n", cudaGetErrorString(err));
  }
}

void fused_gridsample_reduce_mul_h(
  const __half* R,
  const __half* counts,
  __half* slots,
  const int* table,
  int embed_dims,
  int num_cams,
  int H_r,
  int W_r,
  int bev_h,
  int bev_w,
  cudaStream_t stream) {
  const int TX = 128;
  const int QPT = 2;
  const int cols = (bev_w + QPT - 1) / QPT;
  dim3 grid_dim((cols + TX - 1) / TX, bev_h);
  fused_reduce_gather_mul_cuda_kernel_h<2>
    <<<grid_dim, TX, 0, stream>>>(R, table, counts, slots, num_cams, H_r, W_r, bev_h, bev_w, embed_dims);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in fused_gridsample_reduce_mul_cuda_h: %s\n", cudaGetErrorString(err));
  }
}

// void fused_gridsample_reduce_mul_cuda_i8(
//     const int8_t* R,
//     float scaleR,
//     const __half* counts,
//     __half* slots,
//     const int* table,
//     int embed_dims,
//     int num_cams,
//     int H_r,
//     int W_r,
//     int bev_h,
//     int bev_w,
//     cudaStream_t stream) {
//   const int TX = 128;
//   const int QPT = 2;
//   const int cols = (bev_w + QPT - 1) / QPT;
//   dim3 grid_dim((cols + TX - 1) / TX, bev_h);
//   fused_reduce_gather_mul_kernel_i8<2>
//       <<<grid_dim, TX, 0, stream>>>(R, scaleR, table, counts, slots, num_cams, H_r, W_r, bev_h, bev_w, embed_dims);
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     printf("error in fused_gridsample_reduce_mul_cuda_i8: %s\n", cudaGetErrorString(err));
//   }
// }

// void fused_gridsample_reduce_mul_cuda_fp8(
//     const __nv_fp8_e4m3* R,
//     const __half* counts,
//     __half* slots,
//     const int* table,
//     int embed_dims,
//     int num_cams,
//     int H_r,
//     int W_r,
//     int bev_h,
//     int bev_w,
//     cudaStream_t stream) {
//   const int TX = 128;
//   const int QPT = 2;
//   const int cols = (bev_w + QPT - 1) / QPT;
//   dim3 grid_dim((cols + TX - 1) / TX, bev_h);
//   fused_reduce_gather_mul_kernel_fp8<2>
//       <<<grid_dim, TX, 0, stream>>>(R, table, counts, slots, num_cams, H_r, W_r, bev_h, bev_w, embed_dims);
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess) {
//     printf("error in fused_gridsample_reduce_mul_cuda_fp8: %s\n", cudaGetErrorString(err));
//   }
// }