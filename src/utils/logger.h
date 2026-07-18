/**************************************************************
 * @Author: ljw
 * @Date: 2022-03-29 11:43:16
 * @Last Modified by: ljw
 * @Last Modified time: 2022-04-15 15:06:04
 **************************************************************/

#pragma once

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

namespace utils {

// clang-format off
static const char* restore = "\033[0m";  // trace
static const char* blue    = "\033[34m"; // debug
static const char* green   = "\033[32m"; // info
static const char* yellow  = "\033[33m"; // warning
static const char* red     = "\033[31m"; // error
static const char* purple  = "\033[35m"; // fatal
static const char* COLOR_LIST[]{green, blue, restore, yellow, red, purple};
// clang-format on

inline void pretty_printf(
  const int severity,
  const char* const file,
  const int line,
  const char* const func,
  const char* __restrict__ fmt,
  ...) {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  auto now_micros = static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  auto now_seconds = static_cast<time_t>(now_micros / 1000000);
  auto micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  const size_t kTimeBufferSize{30};
  char time_buffer[kTimeBufferSize]{0};
  strftime(time_buffer, kTimeBufferSize, "%Y-%m-%d %H:%M:%S", localtime(&now_seconds));

  const char* p_last_slash = strrchr(file, '/');
  const char* p_base_name = p_last_slash ? p_last_slash + 1 : file;

  const size_t head_len = 256;
  char fixed_head[head_len]{0};
  auto n = snprintf(
    fixed_head,
    head_len,
    "%s.%06d: %c %s:%d] %s ",
    time_buffer,
    micros_remainder,
    "TDIWEF"[severity],
    p_base_name,
    line,
    func);
  if (n < 0) {
    return;
  }

  va_list aptr;
  va_start(aptr, fmt);
  const size_t max_msg_len{1024};
  char user_msg[max_msg_len]{0};
  vsprintf(user_msg, fmt, aptr);
  va_end(aptr);
  fprintf(stdout, "%s%s%s%s\n", fixed_head, COLOR_LIST[severity], user_msg, restore);
  // fprintf(stderr, "%s%s\n", fixed_head, user_msg);
}

inline void log_impl(
  const int lvlid, const char* const file, const char* const func, int line, const char* const fmt, ...) {
  if (lvlid < 2)
    return;
  va_list aptr;
  va_start(aptr, fmt);
  const size_t max_msg_len{1024};
  char user_msg[max_msg_len]{0};
  vsprintf(user_msg, fmt, aptr);
  va_end(aptr);

  pretty_printf(lvlid, file, line, func, user_msg);
}

} // namespace utils

// clang-format off

#define __LOG_PLACEHOLD__     ::utils::log_impl(0, __FILE__, __FUNCTION__, __LINE__, "")
#define LOG_TRACE(fmt, ...) ::utils::log_impl(0, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) ::utils::log_impl(1, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  ::utils::log_impl(2, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__)

#define LOG_WARN(fmt, ...)  ::utils::log_impl(3, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) ::utils::log_impl(4, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_FATAL(fmt, ...) ::utils::log_impl(5, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__)

// clang-format on
