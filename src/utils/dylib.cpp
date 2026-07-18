/**************************************************************
 * RAII wrapper for dlopen/dlsym/dlclose.
 **************************************************************/

#include "dylib.h"

namespace utils {

DynamicLib::DynamicLib(const char* path, int flags) { handle_ = dlopen(path, flags); }

DynamicLib::~DynamicLib() {
  if (handle_) dlclose(handle_);
}

DynamicLib::DynamicLib(DynamicLib&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }

DynamicLib& DynamicLib::operator=(DynamicLib&& other) noexcept {
  if (this != &other) {
    if (handle_) dlclose(handle_);
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

void* DynamicLib::raw_sym(const char* name) const { return handle_ ? dlsym(handle_, name) : nullptr; }

} // namespace utils
