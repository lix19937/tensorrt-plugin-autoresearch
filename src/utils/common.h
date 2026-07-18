#ifndef MYHEADER_H
#define MYHEADER_H

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <sstream>

#include <NvInfer.h>
#include <cuda_runtime.h>

typedef enum {
  STATUS_SUCCESS = 0,
  STATUS_FAILURE = 1,
  STATUS_BAD_PARAM = 2,
  STATUS_NOT_SUPPORTED = 3,
  STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1 {
namespace plugin_custom {

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val) {
  std::memcpy(buffer, &val, sizeof(T));
  buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
T read(const char*& buffer) {
  T val{};
  std::memcpy(&val, buffer, sizeof(T));
  buffer += sizeof(T);
  return val;
}

#define FN_NAME __func__
extern nvinfer1::ILogger* gLogger;
nvinfer1::ILogger* getLogger();

template <ILogger::Severity kSeverity>
class LogStream : public std::ostream {
  class Buf : public std::stringbuf {
   public:
    // int sync() override;
    int sync() override {
      std::string msg = str();
      str("");
      if (gLogger != nullptr) {
        gLogger->log(kSeverity, msg.c_str());
      };
      return 0;
    };
  };
  Buf buffer;
  std::mutex mLogStreamMutex;

 public:
  std::mutex& getMutex() { return mLogStreamMutex; }
  LogStream() : std::ostream(&buffer) {};
};

class TRTException : public std::exception {
 public:
  TRTException(const char* fl, const char* fn, int ln, int st, const char* msg, const char* nm)
    : file(fl), function(fn), line(ln), status(st), message(msg), name(nm) {}

  virtual void log(std::ostream& LogStream) const;
  void setMessage(const char* msg);

 protected:
  const char* file{nullptr};
  const char* function{nullptr};
  int line{0};
  int status{0};
  const char* message{nullptr};
  const char* name{nullptr};
};

class CudaError : public TRTException {
 public:
  CudaError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
    : TRTException(fl, fn, ln, stat, msg, "Cuda") {}
};

class CudnnError : public TRTException {
 public:
  CudnnError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
    : TRTException(fl, fn, ln, stat, msg, "Cudnn") {}
};

class CublasError : public TRTException {
 public:
  CublasError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
    : TRTException(fl, fn, ln, stat, msg, "cuBLAS") {}
};

class PluginError : public TRTException {
 public:
  PluginError(char const* fl, char const* fn, int ln, int stat, char const* msg = nullptr)
    : TRTException(fl, fn, ln, stat, msg, "Plugin") {}
};

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;
extern LogStream<ILogger::Severity::kINFO> gLogInfo;
extern LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

void throwCudaError(const char* file, const char* function, int line, int status, const char* msg);
void reportValidationFailure(char const* msg, char const* file, int line);
void throwPluginError(char const* file, char const* function, int line, int status, char const* msg);
void logError(const char* msg, const char* file, const char* fn, int line);

void reportAssertion(const char* msg, const char* file, int line);

inline void caughtError(const std::exception& e) {
  gLogError << e.what() << std::endl;
}

} // namespace plugin_custom
} // namespace nvinfer1

#define PLUGIN_API_CHECK(condition)                                               \
  {                                                                               \
    if ((condition) == false) {                                                   \
      nvinfer1::plugin_custom::logError(#condition, __FILE__, FN_NAME, __LINE__); \
      return;                                                                     \
    }                                                                             \
  }

#define PLUGIN_API_CHECK_RETVAL(condition, retval)                                \
  {                                                                               \
    if ((condition) == false) {                                                   \
      nvinfer1::plugin_custom::logError(#condition, __FILE__, FN_NAME, __LINE__); \
      return retval;                                                              \
    }                                                                             \
  }

#define PLUGIN_API_CHECK_ENUM_RANGE(Type, val) PLUGIN_API_CHECK(int(val) >= 0 && int(val) < EnumMax<Type>())
#define PLUGIN_API_CHECK_ENUM_RANGE_RETVAL(Type, val, retval) \
  PLUGIN_API_CHECK_RETVAL(int(val) >= 0 && int(val) < EnumMax<Type>(), retval)

#define PLUGIN_CHECK_CUDA(call)  \
  do {                           \
    cudaError_t status = call;   \
    if (status != cudaSuccess) { \
      return status;             \
    }                            \
  } while (0)

#define PLUGIN_CHECK_CUDNN(call)          \
  do {                                    \
    cudnnStatus_t status = call;          \
    if (status != CUDNN_STATUS_SUCCESS) { \
      return status;                      \
    }                                     \
  } while (0)

#define PLUGIN_CUBLASASSERT(status_)                                              \
  {                                                                               \
    auto s_ = status_;                                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                            \
      nvinfer1::plugin_custom::throwCublasError(__FILE__, FN_NAME, __LINE__, s_); \
    }                                                                             \
  }

#define PLUGIN_CUDNNASSERT(status_)                                                   \
  {                                                                                   \
    auto s_ = status_;                                                                \
    if (s_ != CUDNN_STATUS_SUCCESS) {                                                 \
      const char* msg = cudnnGetErrorString(s_);                                      \
      nvinfer1::plugin_custom::throwCudnnError(__FILE__, FN_NAME, __LINE__, s_, msg); \
    }                                                                                 \
  }

#define PLUGIN_CUASSERT(status_)                                                     \
  {                                                                                  \
    auto s_ = status_;                                                               \
    if (s_ != cudaSuccess) {                                                         \
      const char* msg = cudaGetErrorString(s_);                                      \
      printf("%s:%d, cuda error:%d, %s\n", __FILE__, __LINE__, int(s_), msg);        \
      nvinfer1::plugin_custom::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg); \
    }                                                                                \
  }

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define PLUGIN_VALIDATE(...)                                             \
  GET_MACRO(__VA_ARGS__, PLUGIN_VALIDATE_MSG, PLUGIN_VALIDATE_DEFAULT, ) \
  (__VA_ARGS__)

#define PLUGIN_VALIDATE_DEFAULT(condition)                                                   \
  {                                                                                          \
    if (!(condition)) {                                                                      \
      nvinfer1::plugin_custom::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, #condition); \
    }                                                                                        \
  }
#define PLUGIN_VALIDATE_MSG(condition, msg)                                           \
  {                                                                                   \
    if (!(condition)) {                                                               \
      nvinfer1::plugin_custom::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg); \
    }                                                                                 \
  }
// Logs failed assertion and aborts.
// Aborting is undesirable and will be phased-out from the plugin module, at
// which point PLUGIN_ASSERT will perform the same function as PLUGIN_VALIDATE.
#define PLUGIN_ASSERT(assertion)                                                \
  {                                                                             \
    if (!(assertion)) {                                                         \
      printf("fatal error %s:%d\n", __FILE__, __LINE__);                        \
      nvinfer1::plugin_custom::reportAssertion(#assertion, __FILE__, __LINE__); \
    }                                                                           \
  }

#define PLUGIN_FAIL(msg)                                               \
  {                                                                    \
    nvinfer1::plugin_custom::reportAssertion(msg, __FILE__, __LINE__); \
  }

#define PLUGIN_ERROR(msg)                                                           \
  {                                                                                 \
    nvinfer1::plugin_custom::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg); \
  }

#define PLUGIN_CUERROR(status_)                                                             \
  {                                                                                         \
    auto s_ = status_;                                                                      \
    if (s_ != 0)                                                                            \
      nvinfer1::plugin_custom::logError(#status_ " failure.", __FILE__, FN_NAME, __LINE__); \
  }

#endif
