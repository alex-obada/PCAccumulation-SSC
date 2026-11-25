// Module 0: crop + ground removal (libs/dataset.py prep_input), as one stream compaction.
#include <cub/device/device_select.cuh>

#include <stdexcept>

#include "pcacc/config.hpp"
#include "pcacc/modules.hpp"

namespace pcacc {
namespace {

struct KeepPoint {
  __device__ bool operator()(const float4& p) const {
    return fabsf(p.x) < kCropXY && fabsf(p.y) < kCropXY && p.z < kCropZMax && p.z > kCropZMin && p.z > kGroundHeight;
  }
};

}  // namespace

Preprocessor::Preprocessor(int max_points) : max_points_(max_points), d_count_(1) {
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::If(nullptr, bytes, (const float4*)nullptr, (float4*)nullptr, (int*)nullptr,
                                   max_points, KeepPoint{}));
  temp_.reset(bytes);
}

int Preprocessor::run(const float4* raw, int n, float4* out, cudaStream_t s) {
  if (n > max_points_) throw std::runtime_error("Preprocessor: too many points");
  size_t bytes = temp_.size();
  CUDA_CHECK(cub::DeviceSelect::If(temp_.get(), bytes, raw, out, d_count_.get(), n, KeepPoint{}, s));
  return d_count_.download(1, s)[0];
}

}  // namespace pcacc
