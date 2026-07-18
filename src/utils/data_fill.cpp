
/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#include "data_fill.h"

#include <cuda.h>
#include <cuda_fp16.h>

#ifdef DRIVEOS_7030
#include <cuda_fp8.h>
#endif

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>

#include "utils/checkMacros.h"

#include "utils/logger.h"

namespace utils {

template <typename T>
size_t nvsize(T v) {
  return sizeof(T);
}

size_t nvsize(__half v) {
  return 2;
}

#ifdef DRIVEOS_7030
size_t nvsize(__nv_fp8_e4m3 v) {
  return 1;
}
#endif

// declaration function
template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void fill_buffer(void* buffer, int64_t volume, int32_t min, int32_t max);
template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void fill_buffer(void* buffer, int64_t volume, float min, float max);

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type>
void fill_buffer(void* buffer, int64_t volume, int32_t min, int32_t max) {
  T* typedBuffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  std::uniform_int_distribution<int32_t> distribution(min, max);
  auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
  std::generate(typedBuffer, typedBuffer + volume, generator);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type>
void fill_buffer(void* buffer, int64_t volume, float min, float max) {
  T* typedBuffer = static_cast<T*>(buffer);
  std::default_random_engine engine;
  std::uniform_real_distribution<float> distribution(min, max);
  auto generator = [&engine, &distribution]() { return static_cast<T>(distribution(engine)); };
  std::generate(typedBuffer, typedBuffer + volume, generator);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type>
void fill_device_buffer(void*& buffer, int64_t volume, int32_t min, int32_t max, bool with_alloc) {
  T v{};
  auto nbytes = volume * nvsize(v);
  if (with_alloc) {
    CUDA_CHECK(cudaMalloc(&buffer, nbytes));
  }

  void* h_data = malloc(nbytes);
  utils::fill_buffer<T>(h_data, volume, min, max);
  CUDA_CHECK(cudaMemcpy(buffer, h_data, nbytes, cudaMemcpyHostToDevice));
  free(h_data);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type>
void fill_device_buffer(void*& buffer, int64_t volume, float min, float max, bool with_alloc) {
  T v{};
  auto nbytes = volume * nvsize(v);
  if (with_alloc) {
    CUDA_CHECK(cudaMalloc(&buffer, nbytes));
  }

  void* h_data = malloc(nbytes);
  utils::fill_buffer<T>(h_data, volume, min, max);
  CUDA_CHECK(cudaMemcpy(buffer, h_data, nbytes, cudaMemcpyHostToDevice));
  free(h_data);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type>
void fill_host_buffer(void** buffer, int64_t volume, int32_t min, int32_t max, bool with_alloc) {
  T v{};
  auto nbytes = volume * nvsize(v);
  if (with_alloc) {
    *buffer = malloc(nbytes);
  }
  utils::fill_buffer<T>(*buffer, volume, min, max);
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type>
void fill_host_buffer(void** buffer, int64_t volume, float min, float max, bool with_alloc) {
  T v{};
  auto nbytes = volume * nvsize(v);
  if (with_alloc) {
    *buffer = malloc(nbytes);
  }
  utils::fill_buffer<T>(*buffer, volume, min, max);
}

void free_device_buffer(void* buffer) {
  if (buffer != nullptr) {
    CUDA_CHECK(cudaFree(buffer));
  }
}

void free_host_buffer(void* buffer) {
  if (buffer != nullptr) {
    free(buffer);
  }
}

#define FILL_DBUFFER_INSTANCE_INT_IMPL(T) template void fill_device_buffer<T>(void*&, int64_t, int32_t, int32_t, bool);
FILL_DBUFFER_INSTANCE_INT_IMPL(bool)
FILL_DBUFFER_INSTANCE_INT_IMPL(int32_t)
FILL_DBUFFER_INSTANCE_INT_IMPL(int8_t)
FILL_DBUFFER_INSTANCE_INT_IMPL(uint8_t)
FILL_DBUFFER_INSTANCE_INT_IMPL(uint64_t)
FILL_DBUFFER_INSTANCE_INT_IMPL(int64_t)

#define FILL_DBUFFER_INSTANCE_FP_IMPL(T) \
  template void fill_device_buffer<T>(void*& buffer, int64_t volume, float min, float max, bool);
FILL_DBUFFER_INSTANCE_FP_IMPL(float)
FILL_DBUFFER_INSTANCE_FP_IMPL(__half)

#ifdef DRIVEOS_7030
FILL_DBUFFER_INSTANCE_FP_IMPL(__nv_fp8_e4m3)
#endif

#define FILL_HBUFFER_INSTANCE_INT_IMPL(T) \
  template void fill_host_buffer<T>(void** buffer, int64_t volume, int32_t min, int32_t max, bool);
FILL_HBUFFER_INSTANCE_INT_IMPL(bool)
FILL_HBUFFER_INSTANCE_INT_IMPL(int32_t)
FILL_HBUFFER_INSTANCE_INT_IMPL(int8_t)
FILL_HBUFFER_INSTANCE_INT_IMPL(uint8_t)
FILL_HBUFFER_INSTANCE_INT_IMPL(uint64_t)
FILL_HBUFFER_INSTANCE_INT_IMPL(int64_t)

#define FILL_HBUFFER_INSTANCE_FP_IMPL(T) \
  template void fill_host_buffer<T>(void** buffer, int64_t volume, float min, float max, bool);
FILL_HBUFFER_INSTANCE_FP_IMPL(float)
FILL_HBUFFER_INSTANCE_FP_IMPL(__half)
#ifdef DRIVEOS_7030
FILL_HBUFFER_INSTANCE_FP_IMPL(__nv_fp8_e4m3)
#endif

} // namespace utils
