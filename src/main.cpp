/**************************************************************
 * @Author: ljw
 * @Date: 2026-04-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2026-05-15 15:06:04
 **************************************************************/

#include "c_api.h"
#include "utils/cfg_parser.h"
#include "utils/checkMacros.h"
#include "utils/dylib.h"
#include "utils/helper_device.h"
#include "utils/helper_time.h"
#include "utils/helper_trt.h"
#include "utils/ipc.h"
#include "utils/nlohmann/json.hpp"
#include "utils/tensor.h"

#include <NvInfer.h>
#include <cuda.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace config {
constexpr size_t RUN_GROUP_IDX = 0;

constexpr key_t SHM_KEY = 0x8801;
constexpr key_t SEM_KEY = 0x8802;

constexpr const char* PLUGIN_CREATE = "create";
constexpr const char* PLUGIN_ENQUEUE = "enqueue";
constexpr const char* PLUGIN_DESTROY = "destroy";

constexpr size_t SHM_PAYLOAD_ALIGN = 256;

} // namespace config

// Previously-hardcoded tunables, now loaded from a JSON config file (required CLI
// arg). Every field is optional in the JSON: a missing file or missing key falls
// back to the default below, so the harness still runs with an empty/partial file.
struct RunConfig {
  int gpu_device_id = 0;

  float min_latency_decline = 0.2f;
  float min_speedup = 1.2f;

  int default_warmup = 50;
  int default_profile = 1;
  int default_benchmark = 120;
  // The benchmark loop is repeated default_benchmark_reps times; each rep produces
  // one averaged latency, and we report min/median across reps. A single average
  // over one long batch is easily polluted by clock/boost drift and interrupts.
  int default_benchmark_reps = 5;

  // Per-output sanity cap: reject a malformed cfg whose output would demand an
  // absurd shm allocation. NOT the allocated size — shm is sized dynamically.
  size_t each_out_tensor_shm_max_size = 160ull * 1024 * 1024;

  // Path to the io.meta tensor config. No sensible default (it is deployment-
  // specific and required), so it defaults empty and main() rejects an empty/
  // missing path.
  std::string io_meta = "";

  // Plugin .so paths. No sensible default (deployment-specific and required), so
  // they default empty and main() rejects an empty/missing path.
  std::string lib_baseline = "";
  std::string lib_exp = "";
};

// Load RunConfig from JSON. Any missing/unparseable field keeps its default, so a
// missing file just yields the all-defaults config (with a warning). Returns false
// only on hard JSON syntax errors in a file that does exist.
bool load_run_config(const std::string& path, RunConfig& cfg) {
  std::ifstream f(path);
  if (!f) {
    printf(" - warning: config json '%s' not found, using all defaults\n", path.c_str());
    return true;
  }

  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    printf(" - error: failed to parse config json '%s': %s\n", path.c_str(), e.what());
    return false;
  }
  if (!j.is_object()) {
    printf(" - error: config json '%s' top level must be an object\n", path.c_str());
    return false;
  }

  // Pull each key only if present, leaving the default otherwise.
  auto get = [&](const char* key, auto& dst) {
    if (auto it = j.find(key); it != j.end() && !it->is_null()) {
      try {
        dst = it->get<std::decay_t<decltype(dst)>>();
      } catch (const std::exception& e) {
        printf(" - warning: config key '%s' has wrong type, keeping default (%s)\n", key, e.what());
      }
    }
  };
  get("gpu_device_id", cfg.gpu_device_id);
  get("min_latency_decline", cfg.min_latency_decline);
  get("min_speedup", cfg.min_speedup);
  get("default_warmup", cfg.default_warmup);
  get("default_profile", cfg.default_profile);
  get("default_benchmark", cfg.default_benchmark);
  get("default_benchmark_reps", cfg.default_benchmark_reps);
  get("each_out_tensor_shm_max_size", cfg.each_out_tensor_shm_max_size);
  get("io_meta", cfg.io_meta);
  get("lib_baseline", cfg.lib_baseline);
  get("lib_exp", cfg.lib_exp);
  return true;
}

enum class PatternType {
  kGetBaselineLatency = 0,
  kRunExpForNCU,
  kCompareBaselineExpOutputLatency,
  // inner type
  kGetBaselineOutput,
};

enum class Routeype {
  kBaseline = 0,
  kExp,
};

// Shared-memory layout is sized dynamically from the selected io group at runtime:
// both the number of outputs (n_out) and per-output bytes (out_stride) come from
// the cfg, NOT the compile-time NUM_OUTPUT. This keeps the harness correct if a
// plugin/cfg reports a different output count than the header's #define.
//
// Segment = [ ShmHeader | desc[n_out] (POD, flattened) | (align pad) | baseline block | exp block ]
// where each block is n_out * out_stride bytes. The per-output PluginTensorDescs
// live in the segment (they are POD) so the parent can read them back cross-process
// without depending on a fixed-size array in the header.
struct ShmHeader {
  float latency;
  int32_t n_out; // runtime output count
  size_t out_stride; // bytes reserved per output slot (max output size across outputs)
};

constexpr size_t align_up(size_t x, size_t a) {
  return (x + a - 1) / a * a;
}

// PluginTensorDesc array starts right after the header (POD, so it is safe to
// place in shm and memcpy across processes).
inline size_t shm_desc_offset() {
  return align_up(sizeof(ShmHeader), alignof(PluginTensorDesc));
}
inline PluginTensorDesc* shm_descs(void* base) {
  return reinterpret_cast<PluginTensorDesc*>(reinterpret_cast<int8_t*>(base) + shm_desc_offset());
}
// Output payload begins after the desc array, re-aligned for tensor data.
inline size_t shm_payload_offset(size_t n_out) {
  return align_up(shm_desc_offset() + n_out * sizeof(PluginTensorDesc), config::SHM_PAYLOAD_ALIGN);
}
inline size_t shm_total_size(size_t n_out, size_t stride) {
  return shm_payload_offset(n_out) + 2ull * n_out * stride;
}
inline int8_t* shm_baseline(void* base, size_t n_out) {
  return reinterpret_cast<int8_t*>(base) + shm_payload_offset(n_out);
}
inline int8_t* shm_exp(void* base, size_t n_out, size_t stride) {
  return shm_baseline(base, n_out) + n_out * stride;
}

// Bytes needed per output slot for a group (max across outputs, so out[i] lives at
// i*stride). Pure host math on parsed cfg — safe to call before fork().
size_t compute_out_stride(const utils::IoGroup& group) {
  size_t stride = 0;
  for (const auto& t : group.out_tensors) {
    PluginTensorDesc d{};
    d.dims = utils::vec2dims(t.shape);
    d.type = utils::str2dtype(t.dtype);
    stride = std::max(stride, static_cast<size_t>(utils::GetNbBytes(d)));
  }
  return stride;
}

typedef void (*fn_destroy)(void*);
typedef void* (*fn_create)(int, int, const void*, const void*, const void* const*, void* const*, int*, void*);
typedef int (*fn_enqueue)(void*, int, int, const void*, const void*, const void* const*, void* const*, void*, void*);

void worker_run(
  const RunConfig& rc,
  int shmid,
  int semid,
  Routeype worker_type,
  const char* lib_path,
  const char* cfg_path,
  PatternType run_pattern,
  size_t group_idx,
  size_t out_stride) {
  CUDA_CHECK(cudaSetDevice(rc.gpu_device_id));
  utils::DynamicLib lib(lib_path);
  if (!lib) {
    printf(" - error in dlopen %s, iocfg:%s\n", lib_path, cfg_path);
    exit(1);
  }

  fn_create create = lib.sym<fn_create>(config::PLUGIN_CREATE);
  fn_enqueue enqueue = lib.sym<fn_enqueue>(config::PLUGIN_ENQUEUE);
  fn_destroy del = lib.sym<fn_destroy>(config::PLUGIN_DESTROY);
  if (!create || !enqueue || !del) {
    printf(
      " - error: missing symbol(s) in %s (create=%p enqueue=%p destroy=%p)\n",
      lib_path,
      (void*)create,
      (void*)enqueue,
      (void*)del);
    exit(1);
  }

  // ----------------------------------------------------------
  utils::CfgParser p;
  p.load(cfg_path).dedup();
  printf(" - parsed %ld groups, after dedup, %ld groups\n", p.rawCount(), p.size());
  if (p.groups().empty()) {
    printf(" - error in CfgParser, iocfg:%s\n", cfg_path);
    exit(1);
  }
  if (group_idx >= p.size()) {
    printf(" - error: group index %zu out of range (only %zu groups), iocfg:%s\n", group_idx, p.size(), cfg_path);
    exit(1);
  }
  utils::IoGroup group = p.groups()[group_idx];
  const size_t n_in = group.in_tensors.size();
  const size_t n_out = group.out_tensors.size();
  if (n_in == 0 || n_out == 0) {
    printf(" - error: group %zu has n_in=%zu n_out=%zu, iocfg:%s\n", group_idx, n_in, n_out, cfg_path);
    exit(1);
  }

  // ----------------------------------------------------------
  std::vector<trt_edgellm::rt::Tensor> tmp(n_in + n_out);
  std::vector<PluginTensorDesc> inputDescs(n_in);
  std::vector<PluginTensorDesc> outputDescs(n_out);
  std::vector<void*> inputs(n_in);
  std::vector<void*> outputs(n_out);

  auto init = [](PluginTensorDesc& desc, trt_edgellm::rt::Tensor& tensor, const utils::Tensor& cfg_tensor, void*& ptr) {
    desc.dims = utils::vec2dims(cfg_tensor.shape);
    desc.type = utils::str2dtype(cfg_tensor.dtype);
    desc.format = utils::str2layout(cfg_tensor.layout);

    tensor = trt_edgellm::rt::Tensor(desc.dims, trt_edgellm::rt::DeviceType::kGPU, desc.type);
    ptr = tensor.rawPointer();
    // col4 init data (inline values or a .bin file) -> host bytes -> device.
    std::vector<uint8_t> init_bytes = utils::materialize(cfg_tensor);
    if (!init_bytes.empty()) {
      CUDA_CHECK(cudaMemcpy(ptr, init_bytes.data(), init_bytes.size(), cudaMemcpyDefault));
    }
  };
  for (size_t i = 0; i < n_in; ++i) {
    init(inputDescs[i], tmp[i], group.in_tensors[i], inputs[i]);
  }
  for (size_t i = 0; i < n_out; ++i) {
    init(outputDescs[i], tmp[i + n_in], group.out_tensors[i], outputs[i]);
  }

  // CUDA context process isolation requires creating a new TrtCudaStream within the child process
  utils::TrtCudaStream stream;
  utils::TrtCudaEvent start_ev, stop_ev;
  const void* inDescs = reinterpret_cast<const void*>(inputDescs.data());
  const void* outDescs = reinterpret_cast<const void*>(outputDescs.data());
  const void* const* ins = reinterpret_cast<const void* const*>(inputs.data());
  void* const* outs = reinterpret_cast<void* const*>(outputs.data());
  // Initialize to 0: create() may leave this untouched (e.g. workspace-free plugins),
  // and reading an uninitialized value below would be UB.
  int workspace_size = 0;
  void* plug = create(
    static_cast<int>(n_in), static_cast<int>(n_out), inDescs, outDescs, ins, outs, &workspace_size, stream.get());
  // getWorkspaceSize may return 0; a zero-volume Tensor is prohibited, so only
  // allocate when needed and otherwise pass a null workspace to enqueue.
  trt_edgellm::rt::Tensor ws;
  void* ws_ptr = nullptr;
  if (workspace_size > 0) {
    ws = trt_edgellm::rt::Tensor({1, workspace_size}, trt_edgellm::rt::DeviceType::kGPU, DataType::kINT8);
    ws_ptr = ws.rawPointer();
  }

  auto run = [&] {
    enqueue(plug, static_cast<int>(n_in), static_cast<int>(n_out), inDescs, outDescs, ins, outs, ws_ptr, stream.get());
  };
  auto run_sync = [&](int n = 1) {
    for (int i = 0; i < n; ++i) {
      run();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  };

  int warmup_cnt = rc.default_warmup;
  int benchmark_cnt = rc.default_benchmark;
  int profile_cnt = rc.default_profile;
  printf(" - warmup_cnt:%d, profile_cnt:%d, benchmark_cnt:%d\n", warmup_cnt, profile_cnt, benchmark_cnt);
  printf(" - warning: running ncu will affect measured latency!\n");

  std::string latency_pattern{"unknown"};
  bool send{false};
  float latency{-1.f};

  if (PatternType::kGetBaselineLatency == run_pattern) {
    profile_cnt = 0;
    latency_pattern = "baseline_latency";
  } else if (PatternType::kGetBaselineOutput == run_pattern) {
    warmup_cnt = 1;
    profile_cnt = 0;
    benchmark_cnt = 0;
    send = true;
  } else if (PatternType::kRunExpForNCU == run_pattern) {
    profile_cnt = 1 + 1;
    benchmark_cnt = 0;
  } else if (PatternType::kCompareBaselineExpOutputLatency == run_pattern) {
    profile_cnt = 0;
    latency_pattern = "exp_latency";
    send = true;
  }

  if (warmup_cnt > 0) {
    run_sync(warmup_cnt);
  }
  if (profile_cnt > 0) {
    run_sync(profile_cnt);
  }
  if (benchmark_cnt > 0) {
    // Repeat the timed loop several times; each rep yields one per-call average.
    // Report the min (least noise-contaminated) and median across reps.
    const int reps = std::max(1, rc.default_benchmark_reps);
    std::vector<float> samples;
    samples.reserve(reps);
    for (int r = 0; r < reps; ++r) {
      start_ev.record(stream);
      for (int i = 0; i < benchmark_cnt; ++i) {
        run();
      }
      stop_ev.record(stream);
      stop_ev.synchronize();
      samples.push_back((stop_ev - start_ev) / benchmark_cnt);
    }

    std::sort(samples.begin(), samples.end());
    float min_lat = samples.front();
    float median_lat = samples[samples.size() / 2];
    latency = median_lat;

    printf(
      " - %s:%.4f ms, min:%.4f ms, max:%.4f ms over %d reps x %d calls\n",
      latency_pattern.c_str(),
      median_lat,
      min_lat,
      samples.back(),
      reps,
      benchmark_cnt);
  }

  // outs --> shm
  if (send) {
    for (size_t i = 0; i < n_out; ++i) {
      CUDA_CHECK(cudaMemsetAsync(outs[i], 0, utils::GetNbBytes(outputDescs[i]), stream.get()));
    }
    run_sync();

    void* shm_base = utils::attach_shm(shmid);
    ShmHeader* shm = static_cast<ShmHeader*>(shm_base);
    int8_t* ptr =
      worker_type == Routeype::kBaseline ? shm_baseline(shm_base, n_out) : shm_exp(shm_base, n_out, out_stride);

    // Validate slot capacity before entering the critical section, so we never
    // exit() while holding the semaphore (which would deadlock the parent).
    for (size_t i = 0; i < n_out; ++i) {
      const size_t nbytes = utils::GetNbBytes(outputDescs[i]);
      if (nbytes > out_stride) {
        fprintf(stderr, " - error: out[%zu] needs %zu bytes but shm slot is only %zu bytes\n", i, nbytes, out_stride);
        utils::detach_shm(shm_base);
        exit(1);
      }
    }

    {
      utils::SemGuard g(semid); // critical section: publish outputs + meta to shared memory
      // D2H (all copies are ordered on the same stream; sync once after issuing them)
      for (size_t i = 0; i < n_out; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
          ptr + i * out_stride, outs[i], utils::GetNbBytes(outputDescs[i]), cudaMemcpyDefault, stream.get()));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream.get()));

      shm->latency = latency;
      shm->n_out = static_cast<int32_t>(n_out);
      shm->out_stride = out_stride;
      std::memcpy(shm_descs(shm_base), outputDescs.data(), n_out * sizeof(PluginTensorDesc));
    }

    utils::detach_shm(shm_base);
  }

  del(plug);
}

// Compare baseline_out vs exp_out already published in shared memory (host side).
// Returns true if every output tensor is within the per-dtype diff threshold.
bool verify(const ShmHeader* shm) {
  namespace rt = trt_edgellm::rt;
  bool aligned = true;
  void* shm_base = const_cast<ShmHeader*>(shm);
  const size_t n_out = static_cast<size_t>(shm->n_out);
  const size_t stride = shm->out_stride;
  const PluginTensorDesc* descs = shm_descs(shm_base);
  int8_t* base_block = shm_baseline(shm_base, n_out);
  int8_t* exp_block = shm_exp(shm_base, n_out, stride);

  // Accumulate every out's diff into one line and print once at the end, e.g.
  //   out_diff:x,y,z |Y,N,N   (Y = aligned, N = unaligned) instead of one printf per iteration.
  std::string diffs;
  std::string flags;
  char buf[64];
  for (size_t i = 0; i < n_out; ++i) {
    const PluginTensorDesc& d = descs[i];
    const int64_t offset = static_cast<int64_t>(i) * stride;

    // Wrap the shared-memory bytes as CPU tensors (external memory, not owned).
    rt::Tensor base(base_block + offset, d.dims, rt::DeviceType::kCPU, d.type);
    rt::Tensor exp(exp_block + offset, d.dims, rt::DeviceType::kCPU, d.type);

    float diff = rt::Tensor::getAbsDiff(exp, base);
    float thresh = utils::GetDiffThresh(d.type);

    bool ok = !(diff < 0 || diff > thresh);
    if (i) {
      diffs += ",";
      flags += ",";
    }
    snprintf(buf, sizeof(buf), "%.6f", diff);
    diffs += buf;
    flags += ok ? "Y" : "N";
    if (!ok)
      aligned = false;
  }
  printf(" - out_diff:%s | %s\n", diffs.c_str(), flags.c_str());
  return aligned;
}

// Read back the two workers' results from shared memory, compare outputs for
// alignment, and report the exp-vs-baseline latency (baseline is user-provided).
void compare(const RunConfig& rc, const ShmHeader* shm, int semid, float baseline_latency) {
  bool aligned;
  float exp_latency;
  {
    utils::SemGuard g(semid);
    aligned = verify(shm);
    exp_latency = shm->latency;
  }
  printf(aligned ? " - ✅ outputs aligned\n" : " - ❌ error, outputs UNALIGNED\n");

  // latency_diff: baseline - exp
  float latency_diff = baseline_latency - exp_latency;
  float speedup = (exp_latency > 0.f) ? baseline_latency / exp_latency : 0.f;
  printf(
    " - baseline:%.4f ms, exp:%.4f ms, diff:%.4f ms, speedup:%.2fx\n",
    baseline_latency,
    exp_latency,
    latency_diff,
    speedup);
  if (latency_diff >= rc.min_latency_decline || speedup > rc.min_speedup) {
    printf(" - ✅ ↓ %.4f ms\n", latency_diff);
  }
}

// Fork a child that runs one route's worker, wait for it, and report abnormal exits.
// Aborts the whole program if fork/waitpid themselves fail.
// When use_fork is false, run the worker directly in the current process instead
// of forking. This is safe only when a single worker runs for the whole program
// lifetime (e.g. NCU profiling / baseline-latency patterns), where CUDA context
// isolation across routes is not needed.
void fork_worker(
  const RunConfig& rc,
  int shmid,
  int semid,
  Routeype route,
  const char* lib_path,
  const char* cfg_path,
  PatternType pattern,
  size_t group_idx,
  size_t out_stride,
  bool use_fork = true) {
  if (!use_fork) {
    worker_run(rc, shmid, semid, route, lib_path, cfg_path, pattern, group_idx, out_stride);
    return;
  }

  fflush(stdout); // flush before fork so the child doesn't inherit buffered output

  pid_t pid = fork();
  if (pid < 0) {
    perror(" - error, fork failed");
    exit(EXIT_FAILURE);
  }
  if (pid == 0) {
    worker_run(rc, shmid, semid, route, lib_path, cfg_path, pattern, group_idx, out_stride);
    _exit(0); // never fall back to the parent flow (avoids double IPC teardown)
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    perror(" - error, waitpid failed");
    exit(EXIT_FAILURE);
  }
  if (WIFEXITED(status)) {
    int code = WEXITSTATUS(status);
    if (code != 0)
      fprintf(stderr, " - error, worker(route=%d) exited abnormally, code=%d\n", static_cast<int>(route), code);
  } else if (WIFSIGNALED(status)) {
    fprintf(stderr, " - error, worker(route=%d) killed by signal %d\n", static_cast<int>(route), WTERMSIG(status));
  }
}

// ----------------------------------------------------------------------
// Report whether the device's CUDA primary context is already initialized.
// Must read 0 here: the parent must not touch CUDA before fork(), otherwise the
// context leaks into children and their CUDA calls fail (see worker isolation).
// Do NOT call cuInit() to "fix" the query: cuInit itself initializes driver
// state in the parent that does not survive fork and poisons the children.
// If the driver is not initialized yet, that is exactly the clean state we want
// (context inactive), so treat CUDA_ERROR_NOT_INITIALIZED as active == 0.
void check_ctx_inactive(CUdevice dev) {
  unsigned int flags = 0;
  int active = 0;
  CUresult r = cuDevicePrimaryCtxGetState(dev, &flags, &active);
  if (r == CUDA_ERROR_NOT_INITIALIZED) {
    active = 0; // driver not initialized -> primary context definitely inactive
  } else if (r != CUDA_SUCCESS) {
    char const* name = nullptr;
    cuGetErrorName(r, &name);
    fprintf(stderr, " - warning: cuDevicePrimaryCtxGetState failed: %s\n", name ? name : "unknown");
    return;
  }
  printf(" - before fork, ctx active = %d {expected 0}\n", active);
}

// ----------------------------------------------------------------------
void print_help() {
  printf(" - Usage:\n");
  printf("  ./bin config.json 0 [GROUP_IDX]                  Measure baseline latency\n");
  printf("  ./bin config.json 1 [GROUP_IDX]                  Run exp for NCU profiling\n");
  printf("  ./bin config.json 2 BASELINE_LATENCY [GROUP_IDX] Compare output absdiff & measure exp latency\n");
  printf(
    "\n - config.json: harness knobs incl. \"io_meta\" (io.meta path), libs, warmup/benchmark counts, thresholds.\n");
  printf(" - GROUP_IDX: which dedup'd io group to benchmark (default 1)\n");
  printf(" - Example: ./bin config.json 2 0.7500\n");
  printf(" - Example: ./bin config.json 2 0.7500 3\n");
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    print_help();
    return 1;
  }

  auto isFileExist = [](const std::string& filePath) -> bool {
    std::filesystem::path p(filePath);
    return std::filesystem::exists(p) && std::filesystem::is_regular_file(p);
  };

  // argv[1] is the required config.json path. Missing file / missing keys fall
  // back to RunConfig defaults (load returns false only on a real parse error).
  RunConfig rc;
  if (!load_run_config(argv[1], rc)) {
    print_help();
    return 1;
  }

  // io.meta now comes from the config json (key "io_meta"), not the CLI.
  const std::string& iocfg = rc.io_meta;
  if (iocfg.empty() || !isFileExist(iocfg)) {
    printf(" - error: io_meta{%s} from config json not found\n", iocfg.c_str());
    print_help();
    return 1;
  }

  // Plugin .so paths are required (no default); reject empty/missing.
  if (rc.lib_baseline.empty() || !isFileExist(rc.lib_baseline)) {
    printf(" - error: lib_baseline{%s} from config json not found\n", rc.lib_baseline.c_str());
    print_help();
    return 1;
  }
  if (rc.lib_exp.empty() || !isFileExist(rc.lib_exp)) {
    printf(" - error: lib_exp{%s} from config json not found\n", rc.lib_exp.c_str());
    print_help();
    return 1;
  }

  int pattern_raw = 0;
  try {
    pattern_raw = std::stoi(argv[2]);
  } catch (...) {
    printf(" - error: pattern must be integer 0/1/2\n");
    print_help();
    return 1;
  }

  PatternType pattern;
  const char* pattern_desc;
  switch (pattern_raw) {
    case 0:
      pattern = PatternType::kGetBaselineLatency;
      pattern_desc = "current pattern:get baseline latency\n";
      break;
    case 1:
      pattern = PatternType::kRunExpForNCU;
      pattern_desc = "current pattern:run exp for ncu analysis\n";
      break;
    case 2:
      pattern = PatternType::kCompareBaselineExpOutputLatency;
      pattern_desc =
        "current pattern:get baseline output and exp output then compare them,\n"
        "                   get exp latency and compare with baseline latency from user input\n";
      break;
    default:
      printf(" - error: invalid pattern %d, only support 0/1/2\n", pattern_raw);
      print_help();
      return 1;
  }

  float baseline_latency = -1.f;
  // Arg layout: argv[1]=config.json argv[2]=pattern.
  // Pattern 2 puts baseline latency at argv[3], so its optional GROUP_IDX is
  // argv[4]; patterns 0/1 have no baseline arg, so GROUP_IDX is argv[3].
  int group_arg_idx = (pattern == PatternType::kCompareBaselineExpOutputLatency) ? 4 : 3;
  if (pattern == PatternType::kCompareBaselineExpOutputLatency) {
    if (argc < 4) {
      printf(" - error: pattern 2 need baseline latency argument\n");
      print_help();
      return 1;
    }
    try {
      baseline_latency = std::stof(argv[3]);
    } catch (...) {
      printf(" - error: baseline latency must be float number\n");
      return 1;
    }
    printf(" - user input baseline_latency:%.4f ms\n", baseline_latency);
  }

  size_t group_idx = config::RUN_GROUP_IDX;
  if (argc > group_arg_idx) {
    try {
      group_idx = static_cast<size_t>(std::stoul(argv[group_arg_idx]));
    } catch (...) {
      printf(" - error: GROUP_IDX must be a non-negative integer\n");
      print_help();
      return 1;
    }
  }
  printf(" - benchmark group index: %zu\n", group_idx);
  printf(" - %s", pattern_desc);

  const char* cfg_path = iocfg.c_str();

  // ----------------------------------------------------------
  // Parse the cfg in the parent (pure host work, no CUDA) to size the shared
  // memory to the selected group's actual output bytes instead of a fixed 320MB.
  utils::CfgParser parser;
  parser.load(cfg_path).dedup();
  if (group_idx >= parser.size()) {
    printf(" - error: GROUP_IDX %zu out of range (only %zu groups), iocfg:%s\n", group_idx, parser.size(), cfg_path);
    return 1;
  }
  const utils::IoGroup& sel_group = parser.groups()[group_idx];
  const size_t n_out = sel_group.out_tensors.size();
  if (n_out == 0) {
    printf(" - error: group %zu has no output tensors, iocfg:%s\n", group_idx, cfg_path);
    return 1;
  }
  size_t out_stride = compute_out_stride(sel_group);
  if (out_stride == 0 || out_stride > rc.each_out_tensor_shm_max_size) {
    printf(
      " - error: computed out_stride %zu bytes is invalid (cap %zu)\n", out_stride, rc.each_out_tensor_shm_max_size);
    return 1;
  }
  const size_t shm_bytes = shm_total_size(n_out, out_stride);
  printf(
    " - shm sized dynamically: n_out %zu, out_stride %.2f MB, total %.2f MB\n",
    n_out,
    out_stride / 1024.0 / 1024.0,
    shm_bytes / 1024.0 / 1024.0);

  // ----------------------------------------------------------
  check_ctx_inactive(rc.gpu_device_id);

  // -----------------------------------------------------------
  // Each run happens in a forked child so the CUDA context is process-isolated.
  // Outputs/latency are published to shared memory; the parent reads them back.
  utils::SharedMem shm_res(config::SHM_KEY, shm_bytes);
  utils::Semaphore sem_res(config::SEM_KEY, 1);
  const int shmid = shm_res.id();
  const int semid = sem_res.id();
  ShmHeader* shm = static_cast<ShmHeader*>(shm_res.get());
  // Clear the header + desc array; the tensor payload is fully overwritten by workers.
  std::memset(shm, 0, shm_payload_offset(n_out));

  if (pattern == PatternType::kGetBaselineLatency) {
    fork_worker(
      rc,
      shmid,
      semid,
      Routeype::kBaseline,
      rc.lib_baseline.c_str(),
      cfg_path,
      PatternType::kGetBaselineLatency,
      group_idx,
      out_stride,
      /*use_fork=*/false);
  } else if (pattern == PatternType::kRunExpForNCU) {
    fork_worker(
      rc,
      shmid,
      semid,
      Routeype::kExp,
      rc.lib_exp.c_str(),
      cfg_path,
      PatternType::kRunExpForNCU,
      group_idx,
      out_stride,
      /*use_fork=*/false);
  } else { // CompareOutput and Latency
    // 1) baseline child produces the ground-truth output into the baseline block.
    fork_worker(
      rc,
      shmid,
      semid,
      Routeype::kBaseline,
      rc.lib_baseline.c_str(),
      cfg_path,
      PatternType::kGetBaselineOutput,
      group_idx,
      out_stride);

    // 2) exp child produces its output + rt latency into the exp block / header.
    fork_worker(
      rc,
      shmid,
      semid,
      Routeype::kExp,
      rc.lib_exp.c_str(),
      cfg_path,
      PatternType::kCompareBaselineExpOutputLatency,
      group_idx,
      out_stride);

    compare(rc, shm, semid, baseline_latency);
  }

  return 0;
}
