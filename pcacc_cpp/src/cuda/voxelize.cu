// Module 1: points -> 4D pillars (libs/voxel_generator.py, numba on the CPU in the original).
//
//   key[i]  = (t * Ny + y) * Nx + x of point i (kInvalid if outside the grid)
//   sort (key, i) with a radix sort -> points grouped by pillar
//   head[j] = sorted position j starts a new valid pillar; inclusive scan -> pillar id per position
//   scatter: p2v[point] = id, and at each head: coords[id], start[id]
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <stdexcept>

#include "pcacc/config.hpp"
#include "pcacc/modules.hpp"

namespace pcacc {
namespace {

constexpr uint32_t kInvalid = 1u << 19;   // > kNumCells (414720), still inside the sorted bits
constexpr int kKeyBits = 20;
static_assert(kNumCells < (1 << 19), "key does not fit in 19 bits");

__global__ void compute_keys(const float4* __restrict__ pts, int n, uint32_t* __restrict__ keys,
                             int* __restrict__ idx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pts[i];
  // same float32 arithmetic as the numba kernel: floor((p - range_min) / voxel_size)
  int x = static_cast<int>(floorf((p.x - kXMin) / kVx));
  int y = static_cast<int>(floorf((p.y - kYMin) / kVy));
  int z = static_cast<int>(floorf((p.z - kZMin) / kVz));
  int t = static_cast<int>(p.w);
  bool ok = x >= 0 && x < kNx && y >= 0 && y < kNy && z >= 0 && z < kNz && t >= 0 && t < kT;
  keys[i] = ok ? static_cast<uint32_t>((t * kNy + y) * kNx + x) : kInvalid;
  idx[i] = i;
}

__global__ void mark_heads(const uint32_t* __restrict__ keys, int n, int* __restrict__ head) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  uint32_t k = keys[j];
  head[j] = (k != kInvalid && (j == 0 || keys[j - 1] != k)) ? 1 : 0;
}

__global__ void write_pillars(const uint32_t* __restrict__ keys, const int* __restrict__ idx,
                              const int* __restrict__ scan, int n, int* __restrict__ p2v,
                              int4* __restrict__ coords, int* __restrict__ start) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  uint32_t k = keys[j];
  if (k == kInvalid) {
    p2v[idx[j]] = -1;
    if (j == 0 || keys[j - 1] != kInvalid) start[scan[j]] = j;   // end of the last pillar
    return;
  }
  int id = scan[j] - 1;
  p2v[idx[j]] = id;
  if (j == 0 || keys[j - 1] != k) {
    int x = k % kNx, y = (k / kNx) % kNy, t = k / (kNx * kNy);
    coords[id] = make_int4(0, y, x, t);
    start[id] = j;
  }
  if (j == n - 1) start[id + 1] = n;
}

}  // namespace

Voxelizer::Voxelizer(int max_points)
    : max_points_(max_points), keys_(max_points), keys_sorted_(max_points), idx_(max_points),
      idx_sorted_(max_points), head_scan_(max_points), p2v_(max_points), start_(max_points + 1),
      coords_(max_points) {
  size_t sort_bytes = 0, scan_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, (uint32_t*)nullptr, (uint32_t*)nullptr,
                                             (int*)nullptr, (int*)nullptr, max_points, 0, kKeyBits));
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, (int*)nullptr, (int*)nullptr, max_points));
  temp_.reset(std::max(sort_bytes, scan_bytes));
}

Pillars Voxelizer::run(const float4* points, int n, cudaStream_t s) {
  if (n > max_points_) throw std::runtime_error("Voxelizer: too many points");
  Pillars p{0, coords_.get(), p2v_.get(), start_.get(), idx_sorted_.get()};
  if (n == 0) return p;
  const int B = 256, G = div_up(n, B);
  compute_keys<<<G, B, 0, s>>>(points, n, keys_.get(), idx_.get());
  size_t bytes = temp_.size();
  // stable radix sort: inside a pillar the points keep their input order
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp_.get(), bytes, keys_.get(), keys_sorted_.get(), idx_.get(),
                                             idx_sorted_.get(), n, 0, kKeyBits, s));
  mark_heads<<<G, B, 0, s>>>(keys_sorted_.get(), n, head_scan_.get());
  bytes = temp_.size();
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp_.get(), bytes, head_scan_.get(), head_scan_.get(), n, s));
  write_pillars<<<G, B, 0, s>>>(keys_sorted_.get(), idx_sorted_.get(), head_scan_.get(), n, p2v_.get(),
                                coords_.get(), start_.get());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(&p.num, head_scan_.get() + n - 1, sizeof(int), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return p;
}

}  // namespace pcacc
