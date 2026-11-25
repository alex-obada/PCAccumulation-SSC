#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CUDA_CHECK(expr)                                                                      \
  do {                                                                                        \
    cudaError_t err_ = (expr);                                                                \
    if (err_ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, \
                   __LINE__, cudaGetErrorString(err_));                                       \
      std::abort();                                                                           \
    }                                                                                         \
  } while (0)

namespace pcacc {

// Owning device buffer of `size` elements. Allocated once (pipeline startup), never resized.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { reset(n); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept { *this = std::move(o); }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(size_, o.size_);
    return *this;
  }

  void reset(size_t n) {
    release();
    if (n) CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    size_ = n;
  }
  void release() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return size_; }
  size_t bytes() const { return size_ * sizeof(T); }

  void upload(const T* host, size_t n, cudaStream_t s = 0) {
    CUDA_CHECK(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, s));
  }
  std::vector<T> download(size_t n, cudaStream_t s = 0) const {
    std::vector<T> h(n);
    if (n) CUDA_CHECK(cudaMemcpyAsync(h.data(), ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    return h;
  }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

inline int div_up(long long a, int b) { return static_cast<int>((a + b - 1) / b); }

}  // namespace pcacc
