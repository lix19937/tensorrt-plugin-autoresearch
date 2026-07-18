#include "common.h"

namespace nvinfer1 {
namespace plugin_custom {

nvinfer1::ILogger* gLogger = nullptr;
template class LogStream<nvinfer1::ILogger::Severity::kERROR>;
LogStream<nvinfer1::ILogger::Severity::kERROR> gLogError;
nvinfer1::ILogger* getLogger() {
  return gLogger;
}

void TRTException::log(std::ostream& logStream) const {
  logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
  if (message != nullptr) {
    logStream << " (" << message << ")";
  }
  logStream << std::endl;
}

void TRTException::setMessage(const char* msg) {
  message = msg;
}

void throwCudaError(const char* file, const char* function, int line, int status, const char* msg) {
  CudaError error(file, function, line, status, msg);
  error.log(gLogError);
  throw error;
}

void reportAssertion(const char* msg, const char* file, int line) {
  std::ostringstream stream;
  stream << "Assertion failed: " << msg << std::endl << file << ':' << line << std::endl << "Aborting..." << std::endl;
  nvinfer1::ILogger* logger = getLogger();
  if (logger != nullptr) {
    logger->log(ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  }

  // getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
  //                stream.str().c_str());
  PLUGIN_CUASSERT(cudaDeviceReset());
  abort();
}

void reportValidationFailure(char const* msg, char const* file, int line) {
  std::ostringstream stream;
  stream << "Validation failed: " << msg << std::endl << file << ':' << line << std::endl;
  nvinfer1::ILogger* logger = getLogger();
  if (logger != nullptr) {
    logger->log(ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  }
  // getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
  //                stream.str().c_str());
}

void throwPluginError(char const* file, char const* function, int line, int status, char const* msg) {
  PluginError error(file, function, line, status, msg);
  reportValidationFailure(msg, file, line);
  throw error;
}

void logError(const char* msg, const char* file, const char* fn, int line) {
  gLogError << "Parameter check failed at: " << file << "::" << fn << "::" << line;
  gLogError << ", condition: " << msg << std::endl;
}

} // namespace plugin_custom
} // namespace nvinfer1