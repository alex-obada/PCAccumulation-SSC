#pragma once
// The TensorRT parts of the pipeline, each with its fixed tensor names (export/export_onnx.py).
#include <cuda_runtime.h>

#include <string>

#include "pcacc/config.hpp"
#include "pcacc/cuda_utils.hpp"
#include "pcacc/modules.hpp"
#include "pcacc/trt_engine.hpp"

namespace pcacc {

// ---------------------------------------------------------------- module 2: PillarFeatureNet
// pfn_a -> pillar_max + gather_concat -> pfn_b -> pillar_max + gather_concat -> pfn_c -> pillar_max
//   features [N, 9] -> pillar features [M, 32]
class PillarEncoder {
 public:
  PillarEncoder(const std::string& models_dir, Precision p);
  void run(const float* features, int n, const Pillars& pillars, float* pillar_feats, cudaStream_t s);

  // single engine on a [n, in_ch] input -> out [n, 32] (pfn_a: in_ch 9, pfn_b / pfn_c: 64)
  static void run_engine(TrtEngine& e, const float* in, int n, float* out, cudaStream_t s);
  TrtEngine a, b, c;

 private:
  DeviceBuffer<float> net_, cat_, pmax_;
};

// ---------------------------------------------------------------- module 4: UNet + heads
// bev [T, 32, H, W] -> bev_feats [T, 32, H, W], fb_seg [T, 2, H, W], geo_feats [T, 64, H, W]
class Backbone {
 public:
  Backbone(const std::string& models_dir, Precision p);
  void run(const float* bev, float* bev_feats, float* fb_seg, float* geo_feats, cudaStream_t s);
  TrtEngine engine;
};

}  // namespace pcacc
