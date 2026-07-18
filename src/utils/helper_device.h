/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <thread>

#include "checkMacros.h"

namespace utils {

class TrtCudaEvent;

namespace {

void cudaSleep(void* sleep) {
  std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(*static_cast<float*>(sleep)));
}

} // namespace

class TrtCudaStream {
 public:
  TrtCudaStream() {
    CUDA_CHECK(cudaStreamCreate(&mStream)); // cudaStreamCreateWithPriority
  }

  TrtCudaStream(const TrtCudaStream&) = delete;

  TrtCudaStream& operator=(const TrtCudaStream&) = delete;

  TrtCudaStream(TrtCudaStream&&) = delete;

  TrtCudaStream& operator=(TrtCudaStream&&) = delete;

  ~TrtCudaStream() { CUDA_CHECK(cudaStreamDestroy(mStream)); }

  cudaStream_t get() const { return mStream; }

  void synchronize() { CUDA_CHECK(cudaStreamSynchronize(mStream)); }

  void wait(TrtCudaEvent& event);

  void sleep(float* ms) { CUDA_CHECK(cudaLaunchHostFunc(mStream, cudaSleep, ms)); }

 private:
  cudaStream_t mStream{};
};

class TrtCudaEvent {
 public:
  explicit TrtCudaEvent(bool blocking = true) {
    const uint32_t flags = blocking ? cudaEventBlockingSync : cudaEventDefault;
    CUDA_CHECK(cudaEventCreateWithFlags(&mEvent, flags));
  }

  TrtCudaEvent(const TrtCudaEvent&) = delete;

  TrtCudaEvent& operator=(const TrtCudaEvent&) = delete;

  TrtCudaEvent(TrtCudaEvent&&) = delete;

  TrtCudaEvent& operator=(TrtCudaEvent&&) = delete;

  ~TrtCudaEvent() { CUDA_CHECK(cudaEventDestroy(mEvent)); }

  cudaEvent_t get() const { return mEvent; }

  void record(const TrtCudaStream& stream) { CUDA_CHECK(cudaEventRecord(mEvent, stream.get())); }

  void synchronize() { CUDA_CHECK(cudaEventSynchronize(mEvent)); }

  // Returns time elapsed time in milliseconds
  float operator-(const TrtCudaEvent& e) const {
    float time{0};
    CUDA_CHECK(cudaEventElapsedTime(&time, e.get(), get()));
    return time;
  }

 private:
  cudaEvent_t mEvent{};
};

inline void TrtCudaStream::wait(TrtCudaEvent& event) {
  CUDA_CHECK(cudaStreamWaitEvent(mStream, event.get(), 0));
}

class TrtCudaGraph {
 public:
  explicit TrtCudaGraph() = default;

  TrtCudaGraph(const TrtCudaGraph&) = delete;

  TrtCudaGraph& operator=(const TrtCudaGraph&) = delete;

  TrtCudaGraph(TrtCudaGraph&&) = delete;

  TrtCudaGraph& operator=(TrtCudaGraph&&) = delete;

  ~TrtCudaGraph() {
    if (mGraphExec) {
      cudaGraphExecDestroy(mGraphExec);
    }
  }

  void beginCapture(TrtCudaStream& stream) {
    CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
  }

  bool launch(TrtCudaStream& stream) { return cudaGraphLaunch(mGraphExec, stream.get()) == cudaSuccess; }

  void endCapture(TrtCudaStream& stream) {
    CUDA_CHECK(cudaStreamEndCapture(stream.get(), &mGraph));
    CUDA_CHECK(cudaGraphInstantiate(&mGraphExec, mGraph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(mGraph));
  }

  void endCaptureOnError(TrtCudaStream& stream) {
    // There are two possibilities why stream capture would fail:
    // (1) stream is in cudaErrorStreamCaptureInvalidated state.
    // (2) TRT reports a failure.
    // In case (1), the returning mGraph should be nullptr.
    // In case (2), the returning mGraph is not nullptr, but it should not be used.
    const auto ret = cudaStreamEndCapture(stream.get(), &mGraph);
    if (ret == cudaErrorStreamCaptureInvalidated) {
      assert(mGraph == nullptr);
    } else {
      CUDA_CHECK(ret);
      assert(mGraph != nullptr);
      CUDA_CHECK(cudaGraphDestroy(mGraph));
      mGraph = nullptr;
    }
    // Clean up any CUDA error.
    cudaGetLastError();
    std::cout << "The CUDA graph capture on the stream has failed." << std::endl;
  }

 private:
  cudaGraph_t mGraph{};
  cudaGraphExec_t mGraphExec{};
};

} // namespace utils
