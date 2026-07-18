/**************************************************************
 * @Author: lhm & ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#include "c_api.h"
#include <stdio.h>
#include "fused_sca_deform_attn_plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

// user need edit here !!!
#define PLUGIN_NAME FusedSpatialCrossAttentionPluginDynamic

void* create(
  int nbInputs,
  int nbOutputs,
  const void* inputDescs,
  const void* outputDescs,
  const void* const* inputs,
  void* const* outputs,
  int* workspace_size,
  void* stream) {
  auto plugin = new PLUGIN_NAME(); // ----------s0 !!!
  if (plugin->initialize() != 0) { /// ----------s1
    printf("- error in plugin initialize\n");
    delete plugin;
    exit(1);
  }

  plugin->attachToContext(nullptr, nullptr, nullptr); /// ----------s2 !!!

  const PluginTensorDesc* inDescs = reinterpret_cast<const PluginTensorDesc*>(inputDescs);
  const PluginTensorDesc* outDescs = reinterpret_cast<const PluginTensorDesc*>(outputDescs);
  std::vector<DynamicPluginTensorDesc> dynIn;
  dynIn.reserve(nbInputs);
  std::vector<DynamicPluginTensorDesc> dynOut;
  dynOut.reserve(nbOutputs);

  auto toDynamic = [](PluginTensorDesc const& desc) -> DynamicPluginTensorDesc {
    DynamicPluginTensorDesc dyn{};
    dyn.desc = desc;
    dyn.min = desc.dims;
    dyn.max = desc.dims;
    dyn.opt = desc.dims;
    return dyn;
  };
  for (int i = 0; i < nbInputs; ++i) {
    dynIn.push_back(toDynamic(inDescs[i]));
  }
  for (int i = 0; i < nbOutputs; ++i) {
    dynOut.push_back(toDynamic(outDescs[i]));
  }
  plugin->configurePlugin(dynIn.data(), nbInputs, dynOut.data(), nbOutputs); // ----------s3
  *workspace_size = plugin->getWorkspaceSize(inDescs, nbInputs, outDescs, nbOutputs); // ----------s4
  if (*workspace_size < 0) {
    printf(" - error in getWorkspaceSize\n");
    delete plugin;
    exit(1);
  }
  return plugin;
}

int enqueue(
  void* plugin,
  int nbInputs,
  int nbOutputs,
  const void* inputDescs,
  const void* outputDescs,
  const void* const* inputs,
  void* const* outputs,
  void* workspace,
  void* stream) {
  auto p = reinterpret_cast<PLUGIN_NAME*>(plugin);
  const PluginTensorDesc* inDescs = reinterpret_cast<const PluginTensorDesc*>(inputDescs);
  const PluginTensorDesc* outDescs = reinterpret_cast<const PluginTensorDesc*>(outputDescs);
  cudaStream_t custream = reinterpret_cast<cudaStream_t>(stream);

  if (p->enqueue(inDescs, outDescs, inputs, outputs, workspace, custream) != 0) {
    printf(" - error in enqueue\n");
    destroy(plugin);
    exit(1);
  }
  return 0;
}

void destroy(void* plugin) {
  auto p = reinterpret_cast<PLUGIN_NAME*>(plugin);

  p->detachFromContext();
  p->terminate();
  delete p;
}