/*
 * @Author: duyinhe
 * @Date: 2026-07-14 14:29:42
 * @LastEditors: duyinhe
 * @LastEditTime: 2026-07-14 15:28:42
 * @FilePath: /xinxin_plugin/plugin/pillarvfe/vfe_kernel.h
 * @Description:
 *
 * Copyright (c) 2026 by GWM , All Rights Reserved.
 */
#ifndef PILLAR_VFE_KERNEL_H
#define PILLAR_VFE_KERNEL_H
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

template <typename TIn, typename TOut>
int pillarVFELaunch(
    const TIn* features,
    const int* coors,
    const int* num_points,
    const float* W,
    const float* Bias,
    float3 norm,
    float3 vsize,
    float3 offset,
    int maxp,
    float clusterZDiv,
    float centerZDiv,
    int P,
    TOut* out,
    cudaStream_t stream);
#endif
