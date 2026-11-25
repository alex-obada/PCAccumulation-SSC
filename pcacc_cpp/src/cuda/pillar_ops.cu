// Modules 2-3: the non-network parts of PillarFeatureNet (models/pillar_encoder.py) and
// scatter_point_pillar. The MLPs between them are the pfn_a/b/c TensorRT engines.
#include "pcacc/config.hpp"
#include "pcacc/modules.hpp"

namespace pcacc {
namespace {

constexpr int kBlock = 256;

__global__ void pillar_mean_kernel(const float4* __restrict__ pts, const int* __restrict__ start,
                                   const int* __restrict__ sorted_idx, int m, float* __restrict__ mean) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= m) return;
  int b = start[v], e = start[v + 1];
  float sx = 0.f, sy = 0.f, sz = 0.f;
  for (int k = b; k < e; ++k) {
    float4 p = pts[sorted_idx[k]];
    sx += p.x;
    sy += p.y;
    sz += p.z;
  }
  float c = static_cast<float>(e - b);
  mean[3 * v + 0] = sx / c;
  mean[3 * v + 1] = sy / c;
  mean[3 * v + 2] = sz / c;
}

__global__ void point_features_kernel(const float4* __restrict__ pts, int n, const int* __restrict__ p2v,
                                      const int4* __restrict__ coords, const float* __restrict__ mean,
                                      float* __restrict__ f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 p = pts[i];
  float* o = f + static_cast<size_t>(i) * kPfnIn;
  int v = p2v[i];
  if (v < 0) {
    for (int k = 0; k < kPfnIn; ++k) o[k] = 0.f;
    return;
  }
  int4 c = coords[v];   // (z, y, x, t)
  const float x_off = kVx / 2 + kXMin, y_off = kVy / 2 + kYMin;
  float feat[kPfnIn] = {
      p.x, p.y, p.z,
      p.x - mean[3 * v + 0], p.y - mean[3 * v + 1], p.z - mean[3 * v + 2],
      p.x - (c.z * kVx + x_off), p.y - (c.y * kVy + y_off),
      p.w};
#pragma unroll
  for (int k = 0; k < kPfnIn - 1; ++k) o[k] = feat[k] / kPfnScale;
  o[kPfnIn - 1] = feat[kPfnIn - 1] / static_cast<float>(kT);
}

// one thread per (pillar, channel): a warp reads 32 consecutive channels of the same point
__global__ void pillar_max_kernel(const float* __restrict__ net, int ch, const int* __restrict__ start,
                                  const int* __restrict__ sorted_idx, int m, float* __restrict__ out) {
  long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<long long>(m) * ch) return;
  int v = static_cast<int>(t / ch), c = static_cast<int>(t % ch);
  int b = start[v], e = start[v + 1];
  float best = -INFINITY;
  for (int k = b; k < e; ++k) best = fmaxf(best, net[static_cast<size_t>(sorted_idx[k]) * ch + c]);
  out[t] = best;
}

__global__ void gather_concat_kernel(const float* __restrict__ net, const float* __restrict__ pmax, int ch,
                                     int n, const int* __restrict__ p2v, float* __restrict__ out) {
  long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<long long>(n) * 2 * ch) return;
  int i = static_cast<int>(t / (2 * ch)), c = static_cast<int>(t % (2 * ch));
  float val;
  if (c < ch) {
    val = net[static_cast<size_t>(i) * ch + c];
  } else {
    int v = p2v[i];
    val = v < 0 ? 0.f : pmax[static_cast<size_t>(v) * ch + (c - ch)];
  }
  out[t] = val;
}

constexpr int kHW = kNy * kNx;

__global__ void scatter_bev_kernel(const float* __restrict__ feats, const float* __restrict__ mean,
                                   const int4* __restrict__ coords, int m, float* __restrict__ bev,
                                   float* __restrict__ occ, float* __restrict__ mean_map) {
  long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<long long>(m) * kC) return;
  int v = static_cast<int>(t / kC), c = static_cast<int>(t % kC);
  int4 q = coords[v];                       // (z, y, x, t)
  int cell = q.y * kNx + q.z;
  bev[(static_cast<size_t>(q.w) * kC + c) * kHW + cell] = feats[t];
  if (c == 0 && occ) occ[q.w * kHW + cell] = 1.f;
  if (c < 3 && mean_map) mean_map[(static_cast<size_t>(q.w) * 3 + c) * kHW + cell] = mean[3 * v + c];
}

}  // namespace

void pillar_mean(const float4* points, const Pillars& p, float* mean, cudaStream_t s) {
  if (p.num == 0) return;
  pillar_mean_kernel<<<div_up(p.num, kBlock), kBlock, 0, s>>>(points, p.start, p.sorted_idx, p.num, mean);
  CUDA_CHECK(cudaGetLastError());
}

void point_features(const float4* points, int n, const Pillars& p, const float* mean, float* feats,
                    cudaStream_t s) {
  if (n == 0) return;
  point_features_kernel<<<div_up(n, kBlock), kBlock, 0, s>>>(points, n, p.p2v, p.coords, mean, feats);
  CUDA_CHECK(cudaGetLastError());
}

void pillar_max(const float* net, int channels, const Pillars& p, float* out, cudaStream_t s) {
  long long total = static_cast<long long>(p.num) * channels;
  if (total == 0) return;
  pillar_max_kernel<<<div_up(total, kBlock), kBlock, 0, s>>>(net, channels, p.start, p.sorted_idx, p.num, out);
  CUDA_CHECK(cudaGetLastError());
}

void gather_concat(const float* net, const float* pmax, int channels, int n, const Pillars& p, float* out,
                   cudaStream_t s) {
  long long total = static_cast<long long>(n) * 2 * channels;
  if (total == 0) return;
  gather_concat_kernel<<<div_up(total, kBlock), kBlock, 0, s>>>(net, pmax, channels, n, p.p2v, out);
  CUDA_CHECK(cudaGetLastError());
}

void scatter_bev(const float* pillar_feats, const float* mean, const Pillars& p, float* bev, float* occ,
                 float* mean_map, cudaStream_t s) {
  CUDA_CHECK(cudaMemsetAsync(bev, 0, sizeof(float) * kT * kC * kHW, s));
  if (occ) CUDA_CHECK(cudaMemsetAsync(occ, 0, sizeof(float) * kT * kHW, s));
  if (mean_map) CUDA_CHECK(cudaMemsetAsync(mean_map, 0, sizeof(float) * kT * 3 * kHW, s));
  long long total = static_cast<long long>(p.num) * kC;
  if (total == 0) return;
  scatter_bev_kernel<<<div_up(total, kBlock), kBlock, 0, s>>>(pillar_feats, mean, p.coords, p.num, bev, occ,
                                                              mean_map);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace pcacc
