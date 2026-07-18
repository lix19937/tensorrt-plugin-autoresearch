#include "fused_sca_deform_attn_plugin.h"
#include "fused_sca_deform_attn.h"

#include <cuda_fp16.h>
#include <cassert>

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

namespace {
// 6 输入 = 通用 MSDA; 8 输入 = SCA 融合 (6 + restore_bev_grid + counts)。
constexpr int num_in_base = 6;
constexpr int num_in_reduce = 8; // 6 + restore_bev_grid + counts (融合 grid_sample+reduce+mul)
constexpr int num_out = 1;
const char* PLUGIN_VERSION{"1"};

const char* PLUGIN_NAME{"fused_sca_deform_plugin"};

// const char* PLUGIN_NAME{"MultiscaleDeformableAttnPlugin_TRT_HM"};
} // namespace

// Static class fields initialization
PluginFieldCollection FusedSpatialCrossAttentionPluginDynamicCreator::mFC{};
std::vector<PluginField> FusedSpatialCrossAttentionPluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FusedSpatialCrossAttentionPluginDynamicCreator);

FusedSpatialCrossAttentionPluginDynamic::FusedSpatialCrossAttentionPluginDynamic() {}

FusedSpatialCrossAttentionPluginDynamic::FusedSpatialCrossAttentionPluginDynamic(const void* data, size_t length) {
  // Deserialize in the same order as serialization
  const char *d = reinterpret_cast<const char*>(data), *a = d;
  assert(d == a + length);
}

FusedSpatialCrossAttentionPluginDynamic::~FusedSpatialCrossAttentionPluginDynamic() {
  terminate();
}

IPluginV2DynamicExt* FusedSpatialCrossAttentionPluginDynamic::clone() const noexcept {
  try {
    auto* p = new FusedSpatialCrossAttentionPluginDynamic();
    p->setPluginNamespace(mNamespace.c_str());
    p->mBevH = mBevH;
    p->mBevW = mBevW;
    p->mDoReduce = mDoReduce;
    if (0 != p->initialize()) {
      // __LOG_ERROR("FusedSpatialCrossAttentionPluginDynamic init failed\n");
      std::cerr << "FusedSpatialCrossAttentionPluginDynamic init failed " << std::endl;
      return nullptr;
    }
    return p;
  } catch (const std::exception& e) {
    caughtError(e);
    // __LOG_ERROR("%s\n", e.what());
    std::cerr << e.what() << std::endl;
  }
  return nullptr;
}

void FusedSpatialCrossAttentionPluginDynamic::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, nvinfer1::IGpuAllocator* gpuAllocator) noexcept {
  return;
}

void FusedSpatialCrossAttentionPluginDynamic::detachFromContext() noexcept {}

DimsExprs FusedSpatialCrossAttentionPluginDynamic::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept {
  try {
    // 支持 6 输入(通用 MSDA) 或 8 输入(SCA 融合 grid_sample+reduce_sum+mul)。
    assert(nbInputs == num_in_base || nbInputs == num_in_reduce);
    assert(outputIndex < num_out);
    assert(inputs[0].nbDims == 4);
    assert(inputs[1].nbDims == 2);
    assert(inputs[2].nbDims == 1);
    assert(inputs[3].nbDims == 6);
    assert(inputs[4].nbDims == 4);
    assert(inputs[5].nbDims == 5);

    if (nbInputs == num_in_reduce) {
      // 8 输入: restore grid_sample(nearest) + camera reduce_sum(权重全1 conv) +
      // permute + mul_pillarweight 全融合。
      //   inputs[6] = restore_bev_grid [1, H_r, W_r, 2], H_r=num_cams*bev_h, W_r=bev_w
      //   inputs[7] = counts [1, Q, 1], Q = bev_h*bev_w
      // 输出 slots = [1, Q, embed_dims] = [1, 22080, 256]。
      assert(inputs[6].nbDims == 4);
      const int kEmbedDims_c = 256;
      const int kNumCams_c = 7;
      const nvinfer1::IDimensionExpr* h_r = inputs[6].d[1];
      const nvinfer1::IDimensionExpr* w_r = inputs[6].d[2];
      // bev_h = H_r / num_cams; Q = bev_h * bev_w = (H_r/num_cams) * W_r
      const nvinfer1::IDimensionExpr* bev_h =
          exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *h_r, *exprBuilder.constant(kNumCams_c));
      const nvinfer1::IDimensionExpr* Q = exprBuilder.operation(DimensionOperation::kPROD, *bev_h, *w_r);
      nvinfer1::DimsExprs ret;
      ret.nbDims = 3;
      ret.d[0] = exprBuilder.constant(1); // bs = 1
      ret.d[1] = Q; // 22080
      ret.d[2] = exprBuilder.constant(kEmbedDims_c); // 256
      return ret;
    }

    // 6 输入 (通用 MSDA): kernel 直接写融合后的 permute+reshape 布局。
    // 原 plugin 出 [batch=7, num_query=22080, 256] + Python permute/reshape 到
    // [1, 256, 840, 184]; 现由 kernel 直接写该布局 (840=7*120, 22080=120*184)。
    nvinfer1::DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = exprBuilder.constant(1); // bs = 1
    ret.d[1] = exprBuilder.constant(256); // embed_dims = num_heads * channels
    ret.d[2] = exprBuilder.constant(840); // num_cams(7) * grid_h(120)
    ret.d[3] = exprBuilder.constant(184); // grid_w
    return ret;
  } catch (const std::exception& e) {
    // caughtError(e);
  }
  return DimsExprs{};
}

bool FusedSpatialCrossAttentionPluginDynamic::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {
  // 6 输入 (通用 MSDA): value, spatial_shapes, level_start_index, sampling_offsets,
  //   reference_points, attention_weights -> output [1,256,840,184]。
  // 8 输入 (SCA 融合): 再加 restore_bev_grid[1,H_r,W_r,2] + counts[1,Q,1] ->
  //   slots [1,22080,256]。
  assert(nbInputs == num_in_base || nbInputs == num_in_reduce);
  assert(nbOutputs == num_out);
  assert(0 <= pos && pos < nbInputs + nbOutputs);

  switch (pos) {
    case 0: // value: float 或 half, linear
      return (inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    case 1: // spatial_shapes
    case 2: // level_start_index
      return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    case 3: // sampling_offsets
    case 4: // reference_points
    case 5: // attention_weights
      return inOut[pos].type == inOut[0].type && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    case 6: // 6 输入: output; 8 输入: restore_bev_grid
    case 7: // 8 输入: counts
    case 8: // 8 输入: output
      return inOut[pos].type == inOut[0].type && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    default:
      return false;
  }
}

void FusedSpatialCrossAttentionPluginDynamic::configurePlugin(
    const DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const DynamicPluginTensorDesc* outputs,
    int nbOutputs) noexcept {
  // Check for valid input dimensions
  assert(inputs[0].desc.dims.nbDims == 4);
  assert(inputs[1].desc.dims.nbDims == 2);
  assert(inputs[2].desc.dims.nbDims == 1);
  assert(inputs[3].desc.dims.nbDims == 6);
  assert(inputs[4].desc.dims.nbDims == 4);
  assert(inputs[5].desc.dims.nbDims == 5);

  // 8 输入 (SCA 融合): 第 7 路 restore_bev_grid [1,H_r,W_r,2], 第 8 路 counts [1,Q,1]。
  mDoReduce = (nbInputs == num_in_reduce);
  if (mDoReduce) {
    assert(inputs[6].desc.dims.nbDims == 4); // [1, H_r, W_r, 2]
    const int kNumCams = 7;
    const int H_r = inputs[6].desc.dims.d[1]; // = num_cams*bev_h
    const int W_r = inputs[6].desc.dims.d[2]; // = bev_w
    const int bev_h = H_r / kNumCams;
    const int bev_w = W_r;
    mBevH = H_r;
    mBevW = W_r;
    // 预分配常量索引表 (num_cams*bev_h*bev_w ints)。grid 恒定, 表首次 enqueue 惰性构建;
    // 在此 (build 期) 分配, 避免 CUDA-graph capture 内 malloc。
    if (mFusedIndexTable != nullptr) {
      cudaFree(mFusedIndexTable);
      mFusedIndexTable = nullptr;
    }
    const size_t bytes = (size_t)kNumCams * bev_h * bev_w * sizeof(int);
    if (cudaMalloc(&mFusedIndexTable, bytes) == cudaSuccess) {
      mFusedTableBuilt = false;
      mTblCam = kNumCams;
      mTblBevH = bev_h;
      mTblBevW = bev_w;
    }
  }

  // Check M dimensions consistency
  assert(inputs[0].desc.dims.d[2] == inputs[3].desc.dims.d[2]);
  assert(inputs[0].desc.dims.d[2] == inputs[5].desc.dims.d[2]);

  // Check L dimensions consistency
  assert(inputs[1].desc.dims.d[0] == inputs[2].desc.dims.d[0]);
  assert(inputs[1].desc.dims.d[0] == inputs[3].desc.dims.d[3]);
  assert(inputs[1].desc.dims.d[0] == inputs[5].desc.dims.d[3]);

  // Check P dimensions consistency
  assert(inputs[3].desc.dims.d[4] == inputs[5].desc.dims.d[4]);

  // Check Lq dimensions consistency
  assert(inputs[3].desc.dims.d[1] == inputs[5].desc.dims.d[1]);
  // 只支持 channels % 32 == 0 (每线程独占一个 head 通道轴)。
  assert(inputs[0].desc.dims.d[3] % 32 == 0);
}

size_t FusedSpatialCrossAttentionPluginDynamic::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept {
  // 8 输入融合时, msda 核先写 [1,256,840,184] 中间结果到 workspace, 融合核再读它
  // gather+reduce+mul 成最终 slots。需一份该大小 workspace。6 输入无需 workspace。
  if (nbInputs == num_in_reduce) {
    const size_t elems = (size_t)1 * 256 * 840 * 184;
    const size_t elem_sz = (inputs[0].type == nvinfer1::DataType::kHALF) ? 2 : 4;
    return elems * elem_sz;
  }
  return 0;
}

int FusedSpatialCrossAttentionPluginDynamic::enqueue(
    const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  try {
    // [input]  value [batch, spatial_size, num_heads, channels], spatial_shapes,
    //   level_start_index, sampling_offsets, reference_points, attention_weights;
    //   8 输入时再加 restore_bev_grid[1,H_r,W_r,2] + counts[1,Q,1]。
    // [output] 6 输入: [1,256,840,184]; 8 输入: slots[1,22080,256]。
    int batch = inputDesc[0].dims.d[0];
    int spatial_size = inputDesc[0].dims.d[1];
    int num_heads = inputDesc[0].dims.d[2];
    int channels = inputDesc[0].dims.d[3];

    int num_levels = inputDesc[1].dims.d[0];
    int num_query = inputDesc[3].dims.d[1];
    int num_point = inputDesc[3].dims.d[4];
    int points_per_group = inputDesc[4].dims.d[3] / 2;

    // ===== Fused post-plugin permute + reshape =====
    // batch=num_cams=7, num_query=120*184=22080, num_heads*channels=256; kernel 直接写
    // [1, 256, 840, 184] (840=7*120)。用 assert 守护形状。
    const int kNumCams = 7;
    const int kGridH = 120;
    const int kGridW = 184;
    const int kEmbedDims = 256;
    assert(batch == kNumCams && "fused post-plugin reshape expects num_cams=7");
    assert(num_query == kGridH * kGridW && "fused post-plugin reshape expects num_query=22080");
    assert(num_heads * channels == kEmbedDims && "fused post-plugin reshape expects embed_dims=256");
    const int kOutCamsCol = kNumCams * kGridH; // 840

    // 8 输入融合: msda 核先写 [1,256,840,184] 到 workspace, 融合核再 gather+reduce+mul 到
    // outputs[0]。6 输入: msda 核直接写 outputs[0]。
    void* msda_out = outputs[0];
    void* restore_R = nullptr;
    if (mDoReduce) {
      msda_out = workspace;
      restore_R = workspace;
    }

    auto data_type = inputDesc[0].type;
    auto data_type_rp = inputDesc[4].type;
    assert(data_type == DataType::kFLOAT || data_type == DataType::kHALF);
    assert(data_type_rp == DataType::kFLOAT || data_type_rp == DataType::kHALF);

    // 只保留 channels==32 (每线程独占一个 head 通道轴) 的融合路径。
    assert(channels % 32 == 0 && "FusedSpatialCrossAttention only supports channels % 32 == 0");
    switch (data_type) {
      case DataType::kFLOAT:
        sca_deform_attn_f32(
            (float*)inputs[0], // value
            (int32_t*)inputs[1], // spatial_shapes
            (float*)inputs[4], // reference_points
            (float*)inputs[3], // sampling_offsets
            (float*)inputs[5], // attention_weights
            batch,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            points_per_group,
            kOutCamsCol, // 840 = num_cams * grid_h (post-plugin reshape)
            kGridW, // 184 = grid_w
            (float*)msda_out,
            stream);
        break;
      case DataType::kHALF:
        sca_deform_attn_fp16(
            (__half*)inputs[0],
            (int32_t*)inputs[1],
            (__half*)inputs[4],
            (__half*)inputs[3],
            (__half*)inputs[5],
            batch,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            points_per_group,
            kOutCamsCol,
            kGridW,
            (__half*)msda_out,
            stream);
        break;

      default:
        return 1;
    }

    // ===== 8 输入融合: grid_sample(nearest) + camera reduce_sum + permute + mul =====
    // msda 核已把 [1,256,840,184] 写到 workspace; 融合核 gather+reduce+mul, 输出
    // slots[1, 22080, 256] 到 outputs[0]。
    if (mDoReduce) {
      // restore_bev_grid = inputs[6] = [1, H_r, W_r, 2]; counts = inputs[7] = [1, Q, 1]。
      const int H_r = inputDesc[6].dims.d[1]; // 840
      const int W_r = inputDesc[6].dims.d[2]; // 184
      const int bev_h = H_r / kNumCams; // 120
      const int bev_w = W_r; // 184
      const int align_corners = 1; // PyTorch 调用恒为 True
      if (mFusedIndexTable == nullptr) {
        return -1; // configurePlugin 分配失败
      }
      // 索引表每次 enqueue 都在 stream 上重建。
      //
      // 【为什么不能惰性只构建一次】原实现用 mFusedTableBuilt 做"首次构建、后续复用",
      // 在 CUDA graph 下会产生两个致命问题:
      //   1) 构建 kernel 只在第一次 enqueue 跑, 而 TensorRT 的 graph 捕获发生在其后的
      //      某次 enqueue —— 构建 kernel 因此被排除在捕获的 graph 之外, 之后每次 replay
      //      都不会重建表。表内容一旦源于错误的 grid, 永远错下去。
      //   2) 更糟的是, 那"第一次"往往是框架的 warmup / 捕获探测调用, 此时 restore_bev_grid
      //      显存尚未被真实标定数据填充(仍是 0)。全 0 grid 会把所有 (scam,bh,bw) 都算到
      //      同一个像素(如 src=(420,92)), 表整片退化成同一个 base, gather 后每个 query
      //      变成 7 份同一像素之和 —— 这正是"用 cudagraph 结果对不齐"的直接原因。
      //   另外把 build 和 gather 放在两次不同的 enqueue、中间隔着 host 端的
      //   mFusedTableBuilt 判断, 也会让"上一次的错误"在捕获时以
      //   "operation failed due to a previous error during capture" 的形式暴露。
      // 构建核极轻(num_cams*bev_h*bev_w 个线程各一次 round+边界判断), 每次重建的开销
      // 远小于一次错误结果的代价; 且构建与 gather 同在 stream 上顺序执行, 天然被完整
      // 捕获进 graph, replay 时用的永远是当前这次的真实 grid。
      if (data_type == DataType::kFLOAT) {
        fused_reduce_build_index_f(
            (const float*)inputs[6], mFusedIndexTable, kNumCams, H_r, W_r, bev_h, bev_w, align_corners, stream);
      } else {
        fused_reduce_build_index_h(
            (const __half*)inputs[6], mFusedIndexTable, kNumCams, H_r, W_r, bev_h, bev_w, align_corners, stream);
      }
      mFusedTableBuilt = true;
      if (data_type == DataType::kFLOAT) {
        fused_gridsample_reduce_mul_f(
            (const float*)restore_R,
            (const float*)inputs[7],
            (float*)outputs[0],
            mFusedIndexTable,
            kEmbedDims,
            kNumCams,
            H_r,
            W_r,
            bev_h,
            bev_w,
            stream);
      } else { // kHALF
        fused_gridsample_reduce_mul_h(
            (const __half*)restore_R,
            (const __half*)inputs[7],
            (__half*)outputs[0],
            mFusedIndexTable,
            kEmbedDims,
            kNumCams,
            H_r,
            W_r,
            bev_h,
            bev_w,
            stream);
      }
    }

    return 0;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return -1;
}

DataType FusedSpatialCrossAttentionPluginDynamic::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs) const noexcept {
  assert(index < num_out);
  assert(nbInputs == num_in_base || nbInputs == num_in_reduce);
  assert(inputTypes[0] == DataType::kFLOAT || inputTypes[0] == DataType::kHALF);
  return inputTypes[0];
}

const char* FusedSpatialCrossAttentionPluginDynamic::getPluginType() const noexcept {
  return PLUGIN_NAME;
}

const char* FusedSpatialCrossAttentionPluginDynamic::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

int FusedSpatialCrossAttentionPluginDynamic::getNbOutputs() const noexcept {
  return num_out;
}

int FusedSpatialCrossAttentionPluginDynamic::initialize() noexcept {
  return 0;
}

void FusedSpatialCrossAttentionPluginDynamic::terminate() noexcept {
  if (mFusedIndexTable != nullptr) {
    cudaFree(mFusedIndexTable);
    mFusedIndexTable = nullptr;
  }
  mFusedTableBuilt = false;
}

size_t FusedSpatialCrossAttentionPluginDynamic::getSerializationSize() const noexcept {
  return 0;
}

void FusedSpatialCrossAttentionPluginDynamic::serialize(void* buffer) const noexcept {
  char *d = reinterpret_cast<char*>(buffer), *a = d;
  assert(d == a);
}

void FusedSpatialCrossAttentionPluginDynamic::destroy() noexcept {
  delete this;
}

void FusedSpatialCrossAttentionPluginDynamic::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* FusedSpatialCrossAttentionPluginDynamic::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}

/////////////////////////////////////////////////////////

FusedSpatialCrossAttentionPluginDynamicCreator::FusedSpatialCrossAttentionPluginDynamicCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* FusedSpatialCrossAttentionPluginDynamicCreator::getPluginName() const noexcept {
  return PLUGIN_NAME;
}

const char* FusedSpatialCrossAttentionPluginDynamicCreator::getPluginVersion() const noexcept {
  return PLUGIN_VERSION;
}

const PluginFieldCollection* FusedSpatialCrossAttentionPluginDynamicCreator::getFieldNames() noexcept {
  return &mFC;
}

IPluginV2* FusedSpatialCrossAttentionPluginDynamicCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept {
  try {
    auto* plugin = new FusedSpatialCrossAttentionPluginDynamic();
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (const std::exception& e) {
    caughtError(e);
    // __LOG_ERROR("%s\n", e.what());
    std::cerr << e.what() << std::endl;
  }
  return nullptr;
}

IPluginV2* FusedSpatialCrossAttentionPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept {
  try {
    auto* plugin = new FusedSpatialCrossAttentionPluginDynamic(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
  } catch (const std::exception& e) {
    caughtError(e);
  }
  return nullptr;
}

void FusedSpatialCrossAttentionPluginDynamicCreator::setPluginNamespace(const char* libNamespace) noexcept {
  try {
    mNamespace = libNamespace;
  } catch (const std::exception& e) {
    caughtError(e);
  }
}

const char* FusedSpatialCrossAttentionPluginDynamicCreator::getPluginNamespace() const noexcept {
  return mNamespace.c_str();
}
