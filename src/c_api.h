/**************************************************************
 * @Author: ljw
 * @Date: 2026-04-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2026-05-15 15:06:04
 **************************************************************/

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* create(
  int nbInputs,
  int nbOutputs,
  const void* inputDescs, // const PluginTensorDesc*
  const void* outputDescs, // const PluginTensorDesc*
  const void* const* inputs,
  void* const* outputs,
  int* workspace_size,
  void* stream); // cudaStream_t

int enqueue(
  void* plugin,
  int nbInputs,
  int nbOutputs,
  const void* inputDescs, // const PluginTensorDesc*
  const void* outputDescs, // const PluginTensorDesc*
  const void* const* inputs,
  void* const* outputs,
  void* workspace, // user send & mange
  void* stream);

void destroy(void* plugin);

#ifdef __cplusplus
}
#endif
