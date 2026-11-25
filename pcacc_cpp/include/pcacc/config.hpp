#pragma once
// Model / data constants (configs/waymo/waymo.yaml + configs/default.yaml).

namespace pcacc {

// crop + ground removal (libs/dataset.py prep_input)
constexpr float kCropXY = 32.f;                   // |x|, |y| < 32
constexpr float kCropZMin = -2.f, kCropZMax = 6.f;
// ground_height + ground_slack, computed in double then cast, like numpy comparing float32 points
constexpr float kGroundHeight = static_cast<float>(0.04 + 0.3);   // keep z > 0.34

// voxel grid: range [-36, -36, -2, 36, 36, 6], voxel size [0.25, 0.25, 8] -> 288 x 288 x 1 pillars
constexpr float kXMin = -36.f, kYMin = -36.f, kZMin = -2.f;
constexpr float kVx = 0.25f, kVy = 0.25f, kVz = 8.f;
constexpr int kNx = 288, kNy = 288, kNz = 1;
constexpr int kT = 5;                             // sweeps
constexpr int kC = 32;                            // pillar / BEV channels
constexpr int kPfnIn = 9;                         // per-point features
constexpr float kPfnScale = 36.f;                 // |range[0]|: first 8 features are divided by it

constexpr int kNumCells = kT * kNy * kNx;         // max 4D pillars (z has a single cell)
constexpr int kGeoC = 64;                         // backbone geo_feats channels (ego-motion features)
constexpr int kSegC = 2;                          // fb_seg channels (background / foreground)

// TensorRT profiles of the point-wise engines (pfn_a/b/c): N after preprocessing is 26k-608k on the
// selected frames. A fixed range keeps the cached engines valid for every frame set.
constexpr int kMaxPoints = 800000;
constexpr int kPfnOptPoints = 350000;

}  // namespace pcacc
