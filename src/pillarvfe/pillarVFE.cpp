//
// PillarVFEPlugin: fuse PointPillars reader(point-feature-aug) + PFN into one kernel.
//
#include "pillarVFE.h"
#include <cassert>
#include <cstring>
#include <iostream>

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"PillarVFEPlugin"};

PluginFieldCollection PillarVFEPluginCreator::mFC{};
std::vector<PluginField> PillarVFEPluginCreator::mPluginAttributes;

static const int VFE_C = 11;
static const int VFE_OUT = 64;

PillarVFEPlugin::PillarVFEPlugin(const std::vector<float>& W, const std::vector<float>& B,
                                 float3 norm, float3 vsize, float3 offset,
                                 int maxp, float clusterZDiv, float centerZDiv)
  : mW(W), mB(B), mNorm(norm), mVsize(vsize), mOffset(offset),
    mMaxp(maxp), mClusterZDiv(clusterZDiv), mCenterZDiv(centerZDiv) {}

PillarVFEPlugin::PillarVFEPlugin(const void* data, size_t length) {
  const char* d = reinterpret_cast<const char*>(data);
  mW.resize(VFE_OUT * VFE_C);
  mB.resize(VFE_OUT);
  std::memcpy(mW.data(), d, mW.size() * sizeof(float)); d += mW.size() * sizeof(float);
  std::memcpy(mB.data(), d, mB.size() * sizeof(float)); d += mB.size() * sizeof(float);
  std::memcpy(&mNorm,   d, sizeof(float3)); d += sizeof(float3);
  std::memcpy(&mVsize,  d, sizeof(float3)); d += sizeof(float3);
  std::memcpy(&mOffset, d, sizeof(float3)); d += sizeof(float3);
  std::memcpy(&mMaxp,   d, sizeof(int));    d += sizeof(int);
  std::memcpy(&mClusterZDiv, d, sizeof(float)); d += sizeof(float);
  std::memcpy(&mCenterZDiv,  d, sizeof(float)); d += sizeof(float);
}

nvinfer1::IPluginV2DynamicExt* PillarVFEPlugin::clone() const noexcept {
  auto* p = new PillarVFEPlugin(mW, mB, mNorm, mVsize, mOffset, mMaxp, mClusterZDiv, mCenterZDiv);
  p->setPluginNamespace(mNamespace.c_str());
  p->initialize();
  return p;
}

nvinfer1::DimsExprs PillarVFEPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // input0 features [P,20,5] -> output [P,64]
  DimsExprs out;
  out.nbDims = 2;
  out.d[0] = inputs[0].d[0];               // P
  out.d[1] = exprBuilder.constant(VFE_OUT);
  return out;
}

bool PillarVFEPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  // 0: features (fp32 preferred / fp16 linear), 1: coors int32, 2: num_points int32,
  // 3: out (fp16 to feed PPScatter; decoupled from input dtype so fp32-in keeps
  //         full-precision reader math while output stays fp16).
  const PluginTensorDesc& d = inOut[pos];
  if (pos == 0) return (d.type == DataType::kFLOAT || d.type == DataType::kHALF) && d.format == TensorFormat::kLINEAR;
  if (pos == 1) return d.type == DataType::kINT32 && d.format == TensorFormat::kLINEAR;
  if (pos == 2) return d.type == DataType::kINT32 && d.format == TensorFormat::kLINEAR;
  // output is always fp16 (matches getOutputDataType) to feed PPScatter; accepting
  // kFLOAT here would let TRT pick an fp32 output while the buffer is sized fp16 -> OOB.
  if (pos == 3) return d.type == DataType::kHALF && d.format == TensorFormat::kLINEAR;
  return false;
}

void PillarVFEPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
    const nvinfer1::DynamicPluginTensorDesc*, int) noexcept {}

size_t PillarVFEPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
    const nvinfer1::PluginTensorDesc*, int) const noexcept { return 0; }

int PillarVFEPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
  int P = inputDesc[0].dims.d[0];
  auto coors = static_cast<const int*>(inputs[1]);
  auto npts  = static_cast<const int*>(inputs[2]);
  bool inHalf  = inputDesc[0].type == DataType::kHALF;
  bool outHalf = outputDesc[0].type == DataType::kHALF;
  if (!inHalf && outHalf) {          // fp32 in -> fp16 out (recommended)
    return pillarVFELaunch<float, __half>(static_cast<const float*>(inputs[0]), coors, npts,
                                          mWDev, mBDev, mNorm, mVsize, mOffset, mMaxp, mClusterZDiv, mCenterZDiv, P, static_cast<__half*>(outputs[0]), stream);
  } else if (!inHalf && !outHalf) {  // fp32 in -> fp32 out
    return pillarVFELaunch<float, float>(static_cast<const float*>(inputs[0]), coors, npts,
                                         mWDev, mBDev, mNorm, mVsize, mOffset, mMaxp, mClusterZDiv, mCenterZDiv, P, static_cast<float*>(outputs[0]), stream);
  } else {                           // fp16 in -> fp16 out (back-compat)
    return pillarVFELaunch<__half, __half>(static_cast<const __half*>(inputs[0]), coors, npts,
                                           mWDev, mBDev, mNorm, mVsize, mOffset, mMaxp, mClusterZDiv, mCenterZDiv, P, static_cast<__half*>(outputs[0]), stream);
  }
}

nvinfer1::DataType PillarVFEPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {
  // Output fp16 to feed PPScatter regardless of (fp32) input dtype.
  return nvinfer1::DataType::kHALF;
}

const char* PillarVFEPlugin::getPluginType() const noexcept { return PLUGIN_NAME; }
const char* PillarVFEPlugin::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
int PillarVFEPlugin::getNbOutputs() const noexcept { return 1; }

int PillarVFEPlugin::initialize() noexcept {
  if (!mWDev) { cudaMalloc(&mWDev, mW.size()*sizeof(float)); cudaMemcpy(mWDev, mW.data(), mW.size()*sizeof(float), cudaMemcpyHostToDevice); }
  if (!mBDev) { cudaMalloc(&mBDev, mB.size()*sizeof(float)); cudaMemcpy(mBDev, mB.data(), mB.size()*sizeof(float), cudaMemcpyHostToDevice); }
  return 0;
}
void PillarVFEPlugin::terminate() noexcept {
  if (mWDev) { cudaFree(mWDev); mWDev=nullptr; }
  if (mBDev) { cudaFree(mBDev); mBDev=nullptr; }
}
size_t PillarVFEPlugin::getSerializationSize() const noexcept {
  return (mW.size() + mB.size()) * sizeof(float) + 3 * sizeof(float3) + sizeof(int) + 2 * sizeof(float);
}
void PillarVFEPlugin::serialize(void* buffer) const noexcept {
  char* d = reinterpret_cast<char*>(buffer);
  std::memcpy(d, mW.data(), mW.size()*sizeof(float)); d += mW.size()*sizeof(float);
  std::memcpy(d, mB.data(), mB.size()*sizeof(float)); d += mB.size()*sizeof(float);
  std::memcpy(d, &mNorm,   sizeof(float3)); d += sizeof(float3);
  std::memcpy(d, &mVsize,  sizeof(float3)); d += sizeof(float3);
  std::memcpy(d, &mOffset, sizeof(float3)); d += sizeof(float3);
  std::memcpy(d, &mMaxp,   sizeof(int));    d += sizeof(int);
  std::memcpy(d, &mClusterZDiv, sizeof(float)); d += sizeof(float);
  std::memcpy(d, &mCenterZDiv,  sizeof(float)); d += sizeof(float);
}
void PillarVFEPlugin::destroy() noexcept { delete this; }
void PillarVFEPlugin::setPluginNamespace(const char* n) noexcept { mNamespace = n; }
const char* PillarVFEPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// ---- Creator ----
PillarVFEPluginCreator::PillarVFEPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(PluginField("weight", nullptr, PluginFieldType::kFLOAT32, VFE_OUT*VFE_C));
  mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, VFE_OUT));
  mPluginAttributes.emplace_back(PluginField("norm", nullptr, PluginFieldType::kFLOAT32, 3));
  mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 3));
  mPluginAttributes.emplace_back(PluginField("offset", nullptr, PluginFieldType::kFLOAT32, 3));
  mPluginAttributes.emplace_back(PluginField("maxp", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("cluster_z_div", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("center_z_div", nullptr, PluginFieldType::kFLOAT32, 1));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}
const char* PillarVFEPluginCreator::getPluginName() const noexcept { return PLUGIN_NAME; }
const char* PillarVFEPluginCreator::getPluginVersion() const noexcept { return PLUGIN_VERSION; }
const nvinfer1::PluginFieldCollection* PillarVFEPluginCreator::getFieldNames() noexcept { return &mFC; }

IPluginV2DynamicExt* PillarVFEPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {
  std::vector<float> W, B;
  // defaults = AEB values (back-compat with onnx that only bakes weight/bias)
  float3 norm{183.2f, 48.0f, 5.0f}, vsize{0.2f, 0.2f, 8.0f}, offset{-27.9f, -47.9f, 1.0f};
  int maxp=20; float clusterZDiv=1.0f, centerZDiv=1.0f;
  for (int i = 0; i < fc->nbFields; ++i) {
    const PluginField& f = fc->fields[i];
    if (!strcmp(f.name, "weight")) { const float* p=static_cast<const float*>(f.data); W.assign(p, p+f.length); }
    else if (!strcmp(f.name, "bias")) { const float* p=static_cast<const float*>(f.data); B.assign(p, p+f.length); }
    else if (!strcmp(f.name, "norm")) { const float* p=static_cast<const float*>(f.data); norm={p[0],p[1],p[2]}; }
    else if (!strcmp(f.name, "voxel_size")) { const float* p=static_cast<const float*>(f.data); vsize={p[0],p[1],p[2]}; }
    else if (!strcmp(f.name, "offset")) { const float* p=static_cast<const float*>(f.data); offset={p[0],p[1],p[2]}; }
    else if (!strcmp(f.name, "maxp")) { maxp=*static_cast<const int*>(f.data); }
    else if (!strcmp(f.name, "cluster_z_div")) { clusterZDiv=*static_cast<const float*>(f.data); }
    else if (!strcmp(f.name, "center_z_div")) { centerZDiv=*static_cast<const float*>(f.data); }
  }
  auto* plugin = new PillarVFEPlugin(W, B, norm, vsize, offset, maxp, clusterZDiv, centerZDiv);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}
IPluginV2DynamicExt* PillarVFEPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  auto* plugin = new PillarVFEPlugin(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}
void PillarVFEPluginCreator::setPluginNamespace(const char* n) noexcept { mNamespace = n; }
const char* PillarVFEPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

REGISTER_TENSORRT_PLUGIN(PillarVFEPluginCreator);
