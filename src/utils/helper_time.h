/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#include <stdint.h>
#include <sys/time.h>
#include <chrono>

namespace utils {

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

inline TimePoint get_current_time() {
  return std::chrono::high_resolution_clock::now();
}

inline int64_t get_duration(TimePoint end, TimePoint start) {
  return std::chrono::duration<float, std::micro>(end - start).count(); // milli: ms; micro: us
}

inline time_t GetMs() {
  struct timeval time_now{};
  gettimeofday(&time_now, nullptr);
  time_t msecs_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000); // ms
  return msecs_time;
}

inline int64_t GetNs() {
  auto now = std::chrono::high_resolution_clock::now();
  auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  return nanoseconds;
}

} // namespace utils