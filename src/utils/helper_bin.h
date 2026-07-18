/**************************************************************
 * @Author: ljw
 * @Date: 2024-04-12 16:57:59
 * @Last Modified by: ljw
 * @Last Modified time: 2024-04-13 10:14:45
 **************************************************************/

#pragma once

#include <filesystem>
#include <fstream>
#include <type_traits>
#include <vector>

#include "helper_cuda.h"

namespace utils {

inline int mymkdir(const std::string& file_path) {
  std::filesystem::path dir_path = file_path;

  try {
    if (std::filesystem::create_directories(dir_path)) {
    } else {
      if (std::filesystem::exists(dir_path)) {
      } else {
        printf("Error create_directories error, %s\n", file_path.c_str());
        return 1;
      }
    }
  } catch (const std::filesystem::filesystem_error& e) {
    printf("Error:%s, file_path:%s", e.what(), file_path.c_str());
    return 1;
  }

  return 0;
}

template <typename T, typename = std::enable_if_t<!std::is_void_v<T>>>
inline void read_bin(const std::string& file_path, T* output, int num /* ele num */) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    printf("%s, Error open file:%s, num:%d, ple check file/path", __FUNCTION__, file_path.c_str(), num);
    exit(1);
  }

  file.seekg(0, std::ios::end);
  std::streampos file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  printf("exist %d vs expect read %d\n", int(file_size), int(num * sizeof(T)));

  file.read(reinterpret_cast<char*>(output), num * sizeof(T));
  if (!file) {
    printf("Error read file:%s, ple check ele num\n", file_path.c_str());
    exit(1);
  }
  file.close();
}

template <typename T, typename = std::enable_if_t<!std::is_void_v<T>>>
inline void read_bin_to_device(const std::string& file_path, T* output, int num /* ele num */) {
  if (num <= 0) {
    return;
  }

  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    printf("%s, Error open file:%s, num:%d, ple check file/path", __FUNCTION__, file_path.c_str(), num);
    exit(1);
  }

  file.seekg(0, std::ios::end);
  std::streampos file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (int(file_size) != num * int(sizeof(T))) {
    printf(
      "Error read file:%s, exist %d vs expect read {%d * %d} \n",
      file_path.c_str(),
      int(file_size),
      num,
      int(sizeof(T)));
    exit(1);
  }

  std::vector<T> buffer(num);
  file.read(reinterpret_cast<char*>(buffer.data()), num * sizeof(T));
  if (!file) {
    printf("Error read file:%s, ple check element num\n", file_path.c_str());
    exit(1);
  }
  file.close();
  CheckCudaErrors(cudaMemcpy(output, buffer.data(), num * sizeof(T), cudaMemcpyHostToDevice));

  // printf("%s\n", file_path.c_str());
  // for (int i = 0; i < 128 && i < num; ++i) {
  //   if constexpr (std::is_floating_point_v<T>) {
  //     printf("%f, ", buffer[i]);
  //   } else if constexpr (std::is_integral_v<T>) {
  //     printf("%d, ", int(buffer[i]));
  //   } else {
  //     printf("%f, ", float(buffer[i]));
  //   }
  // }
  // printf("\n");
}

template <typename T, typename = std::enable_if_t<!std::is_void_v<T>>>
inline void write_bin(const std::string& file_path, T* input, int num /* ele num */) {
  printf("new a file:%s\n", file_path.c_str());

  auto ensure_parent_dir = [&]() {
    std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());
  };
  ensure_parent_dir();

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    printf("%s, Error open file:%s, ple check file/path\n", __FUNCTION__, file_path.c_str());
    exit(1);
  }

  file.write(reinterpret_cast<const char*>(input), num * sizeof(T));
  file.close();
  printf("done\n");
}

template <typename T, typename = std::enable_if_t<!std::is_void_v<T>>>
inline void write_bin_from_device(const std::string& file_path, T* input, int num /* ele num */) {
  if (num <= 0) {
    return;
  }
  printf("new a file:%s\n", file_path.c_str());
  auto ensure_parent_dir = [&]() {
    std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());
  };
  ensure_parent_dir();

  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    printf("%s, Error open file:%s, ple check file\n", __FUNCTION__, file_path.c_str());
    exit(1);
  }

  std::vector<T> buffer(num);
  CheckCudaErrors(cudaMemcpy(buffer.data(), input, num * sizeof(T), cudaMemcpyDeviceToHost));

  file.write(reinterpret_cast<const char*>(buffer.data()), num * sizeof(T));
  file.close();
}

} // namespace utils
