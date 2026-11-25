#pragma once
// GPU modules of the pipeline. Every module allocates its buffers once, for a maximum number of
// points, and only runs asynchronously on the given stream, except where a size must come back to
// the CPU (N after preprocessing, M after voxelisation): those calls synchronize the stream.
//
// Point layout everywhere: float4 (x, y, z, t) with t = sweep index 0..4 stored as float.
#include <cuda_runtime.h>

#include <cstdint>

#include "pcacc/cuda_utils.hpp"

namespace pcacc {

// ---------------------------------------------------------------- preprocess.cu (module 0)
// Crop |x|, |y| < 32, -2 < z < 6 and ground removal z > 0.34, keeping the input order.
class Preprocessor {
 public:
  explicit Preprocessor(int max_points);
  // raw [n] -> out [N]; returns N (synchronizes).
  int run(const float4* raw, int n, float4* out, cudaStream_t s);

 private:
  int max_points_;
  DeviceBuffer<uint8_t> temp_;
  DeviceBuffer<int> d_count_;
};

// ---------------------------------------------------------------- voxelize.cu (module 1)
// points -> 4D pillars (x, y, t; z has a single cell). Pillars are numbered in increasing order
// of the key (t * Ny + y) * Nx + x, so the numbering differs from numba (first-seen order); the
// set of pillars and the point -> pillar assignment are the same.
struct Pillars {
  int num = 0;              // M
  const int4* coords;       // [M] (z, y, x, t), like `coordinates` in the original
  const int* p2v;           // [N] pillar of each point (-1 if outside the grid)
  const int* start;         // [M + 1] pillar m owns sorted_idx[start[m] .. start[m+1])
  const int* sorted_idx;    // [N] point indices grouped by pillar
};

class Voxelizer {
 public:
  explicit Voxelizer(int max_points);
  // returns the pillar view (synchronizes once to read M).
  Pillars run(const float4* points, int n, cudaStream_t s);

 private:
  int max_points_;
  DeviceBuffer<uint32_t> keys_, keys_sorted_;
  DeviceBuffer<int> idx_, idx_sorted_, head_scan_, p2v_, start_;
  DeviceBuffer<int4> coords_;
  DeviceBuffer<uint8_t> temp_;
};

// ---------------------------------------------------------------- pillar_ops.cu (modules 2-3)
// Per-pillar reductions loop over each pillar's points in sorted order: deterministic, no atomics,
// and correct for negative values (PFN outputs are not post-ReLU).

// mean xyz per pillar [M, 3]
void pillar_mean(const float4* points, const Pillars& p, float* mean, cudaStream_t s);

// the 9 normalized PFN input features per point [N, 9]: x, y, z, distance to pillar mean,
// offset to the pillar centre (x, y), first 8 / 36, then t / 5.
void point_features(const float4* points, int n, const Pillars& p, const float* mean, float* feats,
                    cudaStream_t s);

// max over each pillar's points: net [N, C] -> out [M, C]
void pillar_max(const float* net, int channels, const Pillars& p, float* out, cudaStream_t s);

// out [N, 2C] = cat(net, pmax[p2v]) (input of pfn_b / pfn_c)
void gather_concat(const float* net, const float* pmax, int channels, int n, const Pillars& p, float* out,
                   cudaStream_t s);

// BEV image [T, C, Ny, Nx] (backbone input) + occupancy [T, Ny, Nx] and point-mean map
// [T, 3, Ny, Nx] for ego-motion. Clears the maps first. occ / mean_map may be null.
void scatter_bev(const float* pillar_feats, const float* mean, const Pillars& p, float* bev, float* occ,
                 float* mean_map, cudaStream_t s);

}  // namespace pcacc
