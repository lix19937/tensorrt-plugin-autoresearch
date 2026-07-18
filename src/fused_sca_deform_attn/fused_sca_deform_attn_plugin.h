

#pragma once

#include <vector>
// #include "NvInferPluginCustom.h"
// #include "pluginCustom.h"
#include "common.h"

namespace nvinfer1 {
namespace plugin_custom {

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class FusedSpatialCrossAttentionPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  FusedSpatialCrossAttentionPluginDynamic();

  FusedSpatialCrossAttentionPluginDynamic(const void* data, size_t length);

  // It doesn't make sense to make FusedSpatialCrossAttentionPluginDynamic without arguments,
  // so we delete default constructor.
  // FusedSpatialCrossAttentionPluginDynamic() = delete;

  ~FusedSpatialCrossAttentionPluginDynamic() override;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(
      int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) noexcept override;
  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const noexcept override;
  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

  // IPluginV2 Methods
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  void attachToContext(
      cudnnContext* cudnnContext,
      cublasContext* cublasContext,
      nvinfer1::IGpuAllocator* gpuAllocator) noexcept override;
  void detachFromContext() noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  std::string mNamespace;
  // 8 输入融合模式 (SCA): restore grid_sample(nearest,AC) + camera reduce_sum +
  // permute + mul_pillarweight 全部吃进插件, 输出 slots[1, bev_h*bev_w, embed_dims]
  // = [1, 22080, 256]。第 7 路 restore_bev_grid[1,H_r,W_r,2], 第 8 路 counts[1,Q,1]。
  // mDoReduce=false 时为 6 输入通用 MSDA, 输出 [1,256,840,184]。
  int mBevH = 120; // H_r = num_cams*bev_h
  int mBevW = 184; // W_r = bev_w
  bool mDoReduce = false;
  // 常量索引表 (grid 恒定): 每个 (scam,bh,bw) 的源空间偏移, 越界记 -1。
  // configurePlugin 分配 (避免 graph capture 内 malloc), 首次 enqueue 惰性构建。
  int* mFusedIndexTable = nullptr; // device buffer, num_cams*bev_h*bev_w ints
  bool mFusedTableBuilt = false;
  int mTblCam = 0, mTblBevH = 0, mTblBevW = 0;
};

class FusedSpatialCrossAttentionPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  FusedSpatialCrossAttentionPluginDynamicCreator();

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  const char* getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
} // namespace plugin_custom
} // namespace nvinfer1
