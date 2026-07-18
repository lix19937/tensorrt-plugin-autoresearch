#ifndef PILLAR_VFE_PLUGIN_H
#define PILLAR_VFE_PLUGIN_H
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include "common.h"
#include "vfe_kernel.h"

namespace nvinfer1 {
namespace plugin_custom {

// PillarVFEPlugin: inputs (features[P,20,5], coors[P,4] int32, num_points[P] int32)
// -> output pillar features [P,64]. W[64,11], B[64] baked as attributes.
class PillarVFEPlugin : public nvinfer1::IPluginV2DynamicExt {
 public:
  PillarVFEPlugin() = delete;
  PillarVFEPlugin(const std::vector<float>& W, const std::vector<float>& B,
                  float3 norm, float3 vsize, float3 offset,
                  int maxp, float clusterZDiv, float centerZDiv);
  PillarVFEPlugin(const void* data, size_t length);
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
    int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs, int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  std::string mNamespace;
  std::vector<float> mW;   // [64*11]
  std::vector<float> mB;   // [64]
  float3 mNorm{183.2f, 48.0f, 5.0f};      // xyz normalization divisors (AEB default)
  float3 mVsize{0.2f, 0.2f, 8.0f};        // voxel size xyz
  float3 mOffset{-27.9f, -47.9f, 1.0f};   // range-min offset xyz
  int mMaxp{20};                          // points per pillar (AEB/od:20, ld:10)
  float mClusterZDiv{1.0f};               // extra div on cluster-offset z (ld:8)
  float mCenterZDiv{1.0f};                // extra div on center-offset z  (ld:4)
  float* mWDev{nullptr};
  float* mBDev{nullptr};
};

class PillarVFEPluginCreator : public nvinfer1::IPluginCreator {
 public:
  PillarVFEPluginCreator();
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
  IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;
 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

} // namespace plugin_custom
} // namespace nvinfer1
#endif
