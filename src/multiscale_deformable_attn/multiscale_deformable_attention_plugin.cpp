/**************************************************************
 * @Author: lhm & ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#include "multiscale_deformable_attention_plugin.h"
#include "multiscale_deformable_attention.h"

#include <cuda_fp16.h>

#ifdef DEBUG_PLUGIN
#include "helper_trt.h"
#endif

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

namespace {
constexpr int num_in = 5;
constexpr int num_out = 1;
const char* PLUGIN_VERSION{"1"};

const char* PLUGIN_NAME{"ms_deform_attn_plugin"};

// const char* PLUGIN_NAME{"MultiscaleDeformableAttnPlugin_TRT_HM"};
} // namespace

// Static class fields initialization
PluginFieldCollection MultiscaleDeformAttenPluginDynamicCreator::mFC{};
std::vector<PluginField> MultiscaleDeformAttenPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(MultiscaleDeformAttenPluginDynamicCreator);

MultiscaleDeformAttenPluginDynamic::MultiscaleDeformAttenPluginDynamic() {}

MultiscaleDeformAttenPluginDynamic::MultiscaleDeformAttenPluginDynamic(const void* data, size_t length) {
  // Deserialize in the same order as serialization
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  PLUGIN_ASSERT(d == a + length);
}

MultiscaleDeformAttenPluginDynamic::~MultiscaleDeformAttenPluginDynamic() {
  terminate();
}

IPluginV2DynamicExt* MultiscaleDeformAttenPluginDynamic::clone() const noexcept {
  try {
    auto* p = new MultiscaleDeformAttenPluginDynamic();
    p->setPluginNamespace(mNamespace.c_str());
    if (0 != p->initialize()) {
      printf("MultiscaleDeformAttenPluginDynamic init failed\n");
      return nullptr;
    }
    return p;
  } catch (const std::exception& e) {
    caughtError(e);
    printf("%s\n", e.what());
  }
  return nullptr;
}

void MultiscaleDeformAttenPluginDynamic::attachToContext(
  cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {
  return;
}

void MultiscaleDeformAttenPluginDynamic::detachFromContext() noexcept {}

DimsExprs MultiscaleDeformAttenPluginDynamic::getOutputDimensions(
  int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
  try {
    PLUGIN_ASSERT(nbInputs == num_in);
    PLUGIN_ASSERT(outputIndex < num_out);
    // clang-format off
    // just align to nvself
    PLUGIN_ASSERT(inputs[0].nbDims == 4);
    PLUGIN_ASSERT(inputs[1].nbDims == 2);
    PLUGIN_ASSERT(inputs[2].nbDims == 1);
    PLUGIN_ASSERT(inputs[3].nbDims == 6);
    PLUGIN_ASSERT(inputs[4].nbDims == 5);

    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0]; // batch = value.size(0)
    ret.d[1] = inputs[3].d[1]; // num_query = sampling_locations.size(1)

    // {batch, num_query, num_heads * channels
    int num_heads = inputs[0].d[2]->getConstantValue();
    int channels  = inputs[0].d[3]->getConstantValue();
    // ret.d[2] = exprBuilder.constant(num_heads * channels);
    ret.d[2] = exprBuilder.constant(num_heads);
    ret.d[3] = exprBuilder.constant(channels);
    // clang-format on

    return ret;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return DimsExprs{};
}

bool MultiscaleDeformAttenPluginDynamic::supportsFormatCombination(
  int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  //
  // clang-format off
  // value               torch.Size([2, 81920, 8, 32])          float
  // spatial_shapes 	   torch.Size([4, 2])                     int
  // level_start_index 	 torch.Size([4])                        int
  // sampling_locations  torch.Size([2, 81920, 8, 1, 4, 2])     float
  // attention_weights 	 torch.Size([2, 81920, 8, 1, 4])        float
  // 
  // output	 		         torch.Size([2, 81920, 256])            float
  //                            or torch.Size([2, 81920, 8, 32])
  //
  PLUGIN_ASSERT(nbInputs  == num_in);
  PLUGIN_ASSERT(nbOutputs == num_out);
  PLUGIN_ASSERT(0 <= pos && pos < nbInputs + nbOutputs);

  bool ret1 = false;
  const PluginTensorDesc& desc = inOut[pos];
  switch (pos) {
    case 0: // value
      ret1 = ((desc.type == DataType::kFLOAT || desc.type == DataType::kHALF) && desc.format == TensorFormat::kLINEAR);
      break;
    case 1: // spatial_shapes
    case 2: // level_start_index
      ret1 = (desc.type == DataType::kINT32 && desc.format == TensorFormat::kLINEAR);
      break;
    case 3: // sampling_locations
      ret1 = (desc.type == inOut[0].type && desc.format == TensorFormat::kLINEAR);
      break;
    case 4: // attention_weights
      ret1 = (desc.type == inOut[0].type && desc.format == TensorFormat::kLINEAR);
      break;

    case 5: // output
      ret1 = (desc.type == inOut[0].type && desc.format == TensorFormat::kLINEAR);
      break;

    default:
      break;
  }
  // clang-format on
  return ret1;
}

void MultiscaleDeformAttenPluginDynamic::configurePlugin(
  const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) noexcept {
  // Check for valid input dimensions
  PLUGIN_ASSERT(inputs[0].desc.dims.nbDims == 4);
  PLUGIN_ASSERT(inputs[1].desc.dims.nbDims == 2);
  PLUGIN_ASSERT(inputs[2].desc.dims.nbDims == 1);
  PLUGIN_ASSERT(inputs[3].desc.dims.nbDims == 6);
  PLUGIN_ASSERT(inputs[4].desc.dims.nbDims == 5);

  // Check M dimensions consistency
  PLUGIN_ASSERT(inputs[0].desc.dims.d[2] == inputs[3].desc.dims.d[2]);
  PLUGIN_ASSERT(inputs[0].desc.dims.d[2] == inputs[4].desc.dims.d[2]);

  // Check L dimensions consistency
  PLUGIN_ASSERT(inputs[1].desc.dims.d[0] == inputs[2].desc.dims.d[0]);
  PLUGIN_ASSERT(inputs[1].desc.dims.d[0] == inputs[3].desc.dims.d[3]);
  PLUGIN_ASSERT(inputs[1].desc.dims.d[0] == inputs[4].desc.dims.d[3]);

  // Check P dimensions consistency
  PLUGIN_ASSERT(inputs[3].desc.dims.d[4] == inputs[4].desc.dims.d[4]);

  // Check Lq dimensions consistency
  PLUGIN_ASSERT(inputs[3].desc.dims.d[1] == inputs[4].desc.dims.d[1]);
}

size_t MultiscaleDeformAttenPluginDynamic::getWorkspaceSize(
  const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  return 0;
}

int MultiscaleDeformAttenPluginDynamic::enqueue(
  const PluginTensorDesc* inputDesc,
  const PluginTensorDesc* outputDesc,
  const void* const* inputs,
  void* const* outputs,
  void* workspace,
  cudaStream_t stream) noexcept {
#ifdef DEBUG_PLUGIN
  utils::GetDesc(inputDesc, outputDesc, inputs, num_in, num_out, 16);
  return 0;
#endif

  try {
    // clang-format off
    //  channels = hidden_dim/num_heads
    // input:
    //   value              : float tensor,   dims:[batch, spatial_size, num_heads, channels] 
    //   spatial_shapes 	  : int   tensor,   dims:[number_levels, 2]
    //   level_start_index  : int   tensor,   dims:[number_levels]
    //   sampling_locations : float tensor,   dims:[batch, num_query, num_heads, number_levels, number_point, 2] 
    //   attention_weights  : float tensor,   dims:[batch, num_query, num_heads, number_levels, number_point] 
    // output:
    //                      : float tensor,   dims:[batch, num_query, num_heads * channels]
    //
    int batch        = inputDesc[0].dims.d[0];
    int spatial_size = inputDesc[0].dims.d[1];
    int num_heads    = inputDesc[0].dims.d[2];
    int channels     = inputDesc[0].dims.d[3];

    int num_levels = inputDesc[1].dims.d[0];
    int num_query  = inputDesc[3].dims.d[1];
    int num_point  = inputDesc[3].dims.d[4];
    // clang-format on

    const auto& data_type = inputDesc[0].type;
    PLUGIN_ASSERT(data_type == DataType::kFLOAT || data_type == DataType::kHALF)

    switch (data_type) {
      case DataType::kFLOAT:
        ms_deformable_im2col_cuda(
          stream,
          (float*)inputs[0], // value
          (int*)inputs[1], // spatial_shapes
          (int*)inputs[2], // level_start_index
          (float*)inputs[3], // sampling_locations
          (float*)inputs[4], // attention_weights
          (float*)outputs[0],
          batch,
          spatial_size,
          num_heads,
          channels,
          num_levels,
          num_query,
          num_point);
        break;

      case DataType::kHALF:
        ms_deformable_im2col_cuda(
          stream,
          (half*)inputs[0], // value
          (int*)inputs[1], // spatial_shapes
          (int*)inputs[2], // level_start_index
          (half*)inputs[3], // sampling_locations
          (half*)inputs[4], // attention_weights
          (half*)outputs[0],
          batch,
          spatial_size,
          num_heads,
          channels,
          num_levels,
          num_query,
          num_point);
        break;

      default:
        return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return -1;
}

DataType MultiscaleDeformAttenPluginDynamic::getOutputDataType(
  int index, const DataType* inputTypes, int nbInputs) const noexcept {
  PLUGIN_ASSERT(index < num_out);
  PLUGIN_ASSERT(nbInputs == num_in);
  PLUGIN_ASSERT(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

const char* MultiscaleDeformAttenPluginDynamic::getPluginType() const noexcept {
  return PLUGIN_NAME;
}

const char* MultiscaleDeformAttenPluginDynamic::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

int MultiscaleDeformAttenPluginDynamic::getNbOutputs() const noexcept {
  return num_out;
}

int MultiscaleDeformAttenPluginDynamic::initialize() noexcept {
  return 0;
}

void MultiscaleDeformAttenPluginDynamic::terminate() noexcept {}

size_t MultiscaleDeformAttenPluginDynamic::getSerializationSize() const noexcept {
  return 0;
}

void MultiscaleDeformAttenPluginDynamic::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  PLUGIN_ASSERT(d == a);
}

void MultiscaleDeformAttenPluginDynamic::destroy() noexcept {
  delete this;
}

void MultiscaleDeformAttenPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* MultiscaleDeformAttenPluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

MultiscaleDeformAttenPluginDynamicCreator::MultiscaleDeformAttenPluginDynamicCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* MultiscaleDeformAttenPluginDynamicCreator::getPluginName() const noexcept {
  return PLUGIN_NAME;
}

const char* MultiscaleDeformAttenPluginDynamicCreator::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

const PluginFieldCollection* MultiscaleDeformAttenPluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2* MultiscaleDeformAttenPluginDynamicCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) noexcept {
  try {
    auto* plugin = new MultiscaleDeformAttenPluginDynamic();
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (const std::exception& e) {
    caughtError(e);
    printf("%s\n", e.what());
  }
  return nullptr;
}

IPluginV2* MultiscaleDeformAttenPluginDynamicCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) noexcept {
  try {
    auto* plugin = new MultiscaleDeformAttenPluginDynamic(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return nullptr;
}

void MultiscaleDeformAttenPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* MultiscaleDeformAttenPluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}
