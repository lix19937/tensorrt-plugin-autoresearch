/**************************************************************
 * @Author: lhm & ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#include "c_api.h"
#include <stdio.h>
#include "pillarVFE.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin_custom;

// user need edit here !!!
#define PLUGIN_NAME PillarVFEPlugin

void* create(
  int nbInputs,
  int nbOutputs,
  const void* inputDescs,
  const void* outputDescs,
  const void* const* inputs,
  void* const* outputs,
  int* workspace_size,
  void* stream) {
  // ld ------------- 100000
  // std::vector<float> B{-0.05807996913790703,   -0.0075259036384522915,  0.0037674978375434875,
  // -0.0002147238701581955,
  //                      -0.03215135633945465,   -0.00012853648513555527, 0.10172975063323975, -0.001759763341397047,
  //                      -0.16331274807453156,   -0.000969958258792758,   -0.0006834864616394043,
  //                      -0.022907523438334465, -0.14435304701328278,   -0.002315495628863573,   -0.002576563972979784,
  //                      -0.04180309921503067, -0.001533021917566657,  -0.07588902115821838,    -0.1042410209774971,
  //                      -0.038637347519397736, -0.005958412308245897,  -0.1477852463722229,     -0.16247840225696564,
  //                      -0.001197397243231535, -0.03211041912436485,   -0.27207982540130615,    -0.000323442742228508,
  //                      -0.0006311358883976936, -0.0005297254538163543, -0.001550017623230815, -0.0065297470428049564,
  //                      -0.16895025968551636, -0.023423118516802788,  -0.060151971876621246,   -0.0027406122535467148,
  //                      -0.01572248339653015, -0.0011370759457349777, -0.0007959092035889626,  0.14233872294425964,
  //                      -0.0004380556056275964, -0.0017879237420856953, -0.017333772033452988,   -0.00607785489410162,
  //                      -0.0009210351854562759, -0.0017586636822670698, -0.00028651137836277485,
  //                      -0.0005701985210180283, -0.08065780997276306, -0.0006081284955143929, -0.0003649031277745962,
  //                      -0.22418807446956635,   -0.0001490316935814917, -0.004806742072105408, -0.0005609026993624866,
  //                      -0.004423089325428009,  -0.04886467754840851, -0.0004261927679181099, -0.0002639541635289788,
  //                      -0.0008879750967025757, -0.0002179127186536789, -0.010349765419960022, -0.0008271960541605949,
  //                      -0.000391312874853611,  -0.04428645968437195};
  // std::vector<float> W = B;
  // float clusterZDiv = 4.0f, centerZDiv = 8.0f;
  // int maxp = 10;
  // float3 norm{100, 32, 5}, vsize{0.2f, 0.2f, 8.f}, offset{-27.9f, -47.9f, 1};
  // od ------------- 40000
  std::vector<float> B{
    -0.3900412619113922,  -0.7368246912956238,  -0.8079128861427307,  -0.5902774333953857,  -0.24631953239440918,
    -0.34626442193984985, -0.3208225965499878,  -0.8629615306854248,  -0.2541387975215912,  -0.05166574567556381,
    -0.5999397039413452,  -0.9233020544052124,  0.26846590638160706,  -0.6822057962417603,  -0.12470369040966034,
    -0.2922939956188202,  -0.2021494060754776,  0.05779469013214111,  -0.8893644213676453,  -0.3812359869480133,
    -1.540971040725708,   -0.6172891855239868,  -0.13662657141685486, -0.16142283380031586, -0.4641421437263489,
    -0.2632676362991333,  -0.0816396176815033,  -0.31537267565727234, -0.34361469745635986, -0.0049452632665634155,
    0.7588780522346497,   -0.44944465160369873, -0.4528326392173767,  -0.30188894271850586, -0.2014252245426178,
    -0.5072730779647827,  -0.5300394892692566,  -0.9658558368682861,  -0.06147761642932892, -0.8882482051849365,
    -0.16105329990386963, -0.03776481747627258, -0.6449341773986816,  -0.17264702916145325, -0.5561009049415588,
    -0.49045658111572266, -0.6066696643829346,  -0.283523827791214,   -0.0470559298992157,  -1.5558428764343262,
    -0.07320691645145416, -0.283958375453949,   -0.20255382359027863, -0.3185878396034241,  -0.7866226434707642,
    0.30284684896469116,  -0.18089105188846588, -1.4597876071929932,  -0.6236951351165771,  -0.8533517122268677,
    -0.12294474244117737, -0.3350968062877655,  -0.02543547749519348, -0.1768106371164322};
  std::vector<float> W = B;
  float clusterZDiv = 1.0f, centerZDiv = 1.0f;
  int maxp = 20;
  float3 norm{151.2f, 48, 5}, vsize{0.2f, 0.2f, 8.f}, offset{-27.9f, -47.9f, 1};
  // ---------------------------------------

  auto* plugin = new PLUGIN_NAME(W, B, norm, vsize, offset, maxp, clusterZDiv, centerZDiv); // ----------s0 !!!
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