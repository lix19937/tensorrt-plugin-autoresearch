#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---- Fused spatial-cross-attention deformable attention (32 channels/thread) ----
// 融合 reference_points + sampling_offsets、online softmax、bilinear gather，并把
// Python 端的 post-plugin permute+reshape 直接写成 [1, embed_dims, out_cols, grid_w]
// = [1, 256, 840, 184] 布局 (out_cols = num_cams*grid_h = 840, grid_w = 184)。
// value 形状 [num_cams, spatial, num_heads, channels], channels==32 (每线程独占一个
// head 的整条通道轴，softmax/offset/ref-point 前导开销在通道维零重复)。
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
    const int out_cols, // 840 = num_cams * grid_h
    const int grid_w, // 184 = grid_w
    float* data_col,
    cudaStream_t stream);

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
    const int out_cols, // 840
    const int grid_w, // 184
    __half* data_col,
    cudaStream_t stream);

// ---- Fused restore grid_sample(nearest,AC) + camera reduce_sum + permute + mul ----
// 把 restore_outputs 整段 (grid_sample + query_reduce_sum(权重全1 conv) +
// flatten/permute + mul_pillarweight) 融合成一个核, 直接输出 slots[1,Q,embed_dims]。
//   R      = msda 输出 [1, embed_dims, H_r, W_r] = [1, 256, 840, 184]
//   grid   = restore_bev_grid [1, H_r, W_r, 2] = [1, 840, 184, 2]
//   counts = [1, Q, 1] = [1, 22080, 1], Q = bev_h*bev_w
//   slots  = [1, Q, embed_dims] = [1, 22080, 256]
//   table  = 常量索引表 (num_cams*bev_h*bev_w ints), 由 grid 预计算 (越界记 -1)。
//   align_corners 恒为 True (PyTorch 调用固定), 越界 zeros padding。
void fused_reduce_build_index_f(
    const float* grid,
    int* table,
    int num_cams,
    int H_r,
    int W_r,
    int bev_h,
    int bev_w,
    int align_corners,
    cudaStream_t stream);

void fused_reduce_build_index_h(
    const __half* grid,
    int* table,
    int num_cams,
    int H_r,
    int W_r,
    int bev_h,
    int bev_w,
    int align_corners,
    cudaStream_t stream);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

// int8 R (对称量化, 真实值 = q*scaleR); fp8 e4m3 R (直接编码)。低精度只在生产
// 平滑 grid 下 (相邻 query 读相邻源像素, 读合并) 才省读带宽; 随机 grid 无收益。
void fused_gridsample_reduce_mul_i8(
    const int8_t* R,
    float scaleR,
    const __half* counts,
    __half* slots,
    const int* table,
    int embed_dims,
    int num_cams,
    int H_r,
    int W_r,
    int bev_h,
    int bev_w,
    cudaStream_t stream);

void fused_gridsample_reduce_mul_fp8(
    const __nv_fp8_e4m3* R,
    const __half* counts,
    __half* slots,
    const int* table,
    int embed_dims,
    int num_cams,
    int H_r,
    int W_r,
    int bev_h,
    int bev_w,
    cudaStream_t stream);
