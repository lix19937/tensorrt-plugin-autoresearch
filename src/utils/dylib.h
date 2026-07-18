/**************************************************************
 * RAII wrapper for dlopen/dlsym/dlclose.
 **************************************************************/

#pragma once

#include <dlfcn.h>

namespace utils {

// RAII shared library: dlopen on construction, dlclose on destruction.
// Move-only; the handle is closed exactly once by the final owner.
class DynamicLib {
 public:
  // Open the library; returns a handle-less (falsy) object on failure.
  explicit DynamicLib(const char* path, int flags = RTLD_LAZY | RTLD_LOCAL);
  ~DynamicLib();

  DynamicLib(const DynamicLib&) = delete;
  DynamicLib& operator=(const DynamicLib&) = delete;
  DynamicLib(DynamicLib&& other) noexcept;
  DynamicLib& operator=(DynamicLib&& other) noexcept;

  // Resolve a symbol and cast it to the requested (function) pointer type.
  // Returns nullptr if the symbol is not found.
  template <typename T>
  T sym(const char* name) const {
    return reinterpret_cast<T>(raw_sym(name));
  }

  void* handle() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }

 private:
  void* raw_sym(const char* name) const;
  void* handle_;
};

} // namespace utils
