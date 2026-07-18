
/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#pragma once

#include <stdint.h>
#include <type_traits>

namespace utils {

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void fill_device_buffer(void*& buffer, int64_t volume, int32_t min, int32_t max, bool with_alloc = false);
template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void fill_device_buffer(void*& buffer, int64_t volume, float min, float max, bool with_alloc = false);

void free_device_buffer(void* buffer);

template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void fill_host_buffer(void** buffer, int64_t volume, int32_t min, int32_t max, bool with_alloc = false);
template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void fill_host_buffer(void** buffer, int64_t volume, float min, float max, bool with_alloc = false);

void free_host_buffer(void* buffer);

} // namespace utils
