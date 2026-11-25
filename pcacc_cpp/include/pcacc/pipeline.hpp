#pragma once
// Module order, GPU buffers (allocated once, at the maximum frame size) and the CUDA stream.
//
// Implemented so far: h2d -> preprocess -> voxelize -> pillar_enc (PFN features, pfn_a/b/c engines,
// BEV scatter) -> backbone. Without TensorRT only the CUDA stages up to the PFN features run.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pcacc/cuda_utils.hpp"
#include "pcacc/modules.hpp"
#include "pcacc/pointcloud_io.hpp"
#ifdef PCACC_WITH_TENSORRT
#include "pcacc/networks.hpp"
#endif

namespace pcacc {

struct StageTime {
  std::string stage;   // same names as benchmarks/bench_v0.py
  float ms;
};

struct FrameResult {
  int n_raw = 0, n_points = 0, n_pillars = 0;
  std::vector<StageTime> times;
  float e2e_ms() const;
};

class Pipeline {
 public:
#ifdef PCACC_WITH_TENSORRT
  Pipeline(int max_raw_points, const std::string& models_dir, Precision precision);
#else
  explicit Pipeline(int max_raw_points);
#endif
  ~Pipeline();
  FrameResult run(const Frame& f);

  // device outputs of the last run (valid until the next one)
  const float4* points() const { return points_.get(); }
  const Pillars& pillars() const { return pillars_; }
  const float* pfn_features() const { return feats_.get(); }
#ifdef PCACC_WITH_TENSORRT
  const float* pillar_feats() const { return pillar_feats_.get(); }   // [M, 32]
  const float* bev() const { return bev_.get(); }                      // [T, 32, H, W]
  const float* bev_feats() const { return bev_feats_.get(); }          // [T, 32, H, W]
  const float* fb_seg() const { return fb_seg_.get(); }                // [T, 2, H, W]
  const float* geo_feats() const { return geo_feats_.get(); }          // [T, 64, H, W]
  const float* occupancy() const { return occ_.get(); }                // [T, H, W]
  const float* mean_map() const { return mean_map_.get(); }            // [T, 3, H, W]
#endif

 private:
  class Timer;
  int max_points_;
  cudaStream_t stream_{};
  DeviceBuffer<float4> raw_, points_;
  DeviceBuffer<float> mean_, feats_;
  Preprocessor pre_;
  Voxelizer vox_;
  Pillars pillars_{};
#ifdef PCACC_WITH_TENSORRT
  DeviceBuffer<float> pillar_feats_, bev_, occ_, mean_map_, bev_feats_, fb_seg_, geo_feats_;
  std::unique_ptr<PillarEncoder> pfn_;
  std::unique_ptr<Backbone> backbone_;
#endif
};

}  // namespace pcacc
