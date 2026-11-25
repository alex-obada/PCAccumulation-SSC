#include "pcacc/pipeline.hpp"

#include <algorithm>
#include <stdexcept>

#include "pcacc/config.hpp"

namespace pcacc {

float FrameResult::e2e_ms() const {
  float s = 0.f;
  for (auto& t : times) s += t.ms;
  return s;
}

// cudaEvent pairs on the pipeline stream, read once at the end of the frame
class Pipeline::Timer {
 public:
  explicit Timer(cudaStream_t s) : s_(s) {}
  ~Timer() {
    for (auto& m : marks_) cudaEventDestroy(m.second);
  }
  void mark(const char* stage) {
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(e, s_));
    marks_.emplace_back(stage, e);
  }
  std::vector<StageTime> read() {
    CUDA_CHECK(cudaStreamSynchronize(s_));
    std::vector<StageTime> out;
    for (size_t i = 1; i < marks_.size(); ++i) {
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, marks_[i - 1].second, marks_[i].second));
      out.push_back({marks_[i].first, ms});
    }
    return out;
  }

 private:
  cudaStream_t s_;
  std::vector<std::pair<std::string, cudaEvent_t>> marks_;
};

#ifdef PCACC_WITH_TENSORRT
Pipeline::Pipeline(int max_raw_points, const std::string& models_dir, Precision precision)
#else
Pipeline::Pipeline(int max_raw_points)
#endif
    : max_points_(max_raw_points), raw_(max_raw_points), points_(max_raw_points), mean_(size_t(max_raw_points) * 3),
      feats_(size_t(max_raw_points) * kPfnIn), pre_(max_raw_points), vox_(max_raw_points) {
  CUDA_CHECK(cudaStreamCreate(&stream_));
#ifdef PCACC_WITH_TENSORRT
  constexpr size_t hw = size_t(kNy) * kNx;
  const size_t max_pillars = std::min<size_t>(max_raw_points, kNumCells);
  pillar_feats_.reset(max_pillars * kC);
  bev_.reset(kT * kC * hw);
  occ_.reset(kT * hw);
  mean_map_.reset(kT * 3 * hw);
  bev_feats_.reset(kT * kC * hw);
  fb_seg_.reset(kT * kSegC * hw);
  geo_feats_.reset(kT * kGeoC * hw);
  pfn_ = std::make_unique<PillarEncoder>(models_dir, precision);
  backbone_ = std::make_unique<Backbone>(models_dir, precision);
#endif
}

Pipeline::~Pipeline() { cudaStreamDestroy(stream_); }

FrameResult Pipeline::run(const Frame& f) {
  FrameResult r;
  r.n_raw = static_cast<int>(f.num_points());
  if (r.n_raw > max_points_) throw std::runtime_error("frame larger than the pipeline capacity");
  Timer tm(stream_);
  tm.mark("start");

  raw_.upload(reinterpret_cast<const float4*>(f.xyzt.data()), r.n_raw, stream_);
  tm.mark("h2d");

  r.n_points = pre_.run(raw_.get(), r.n_raw, points_.get(), stream_);
  tm.mark("preprocess");

  pillars_ = vox_.run(points_.get(), r.n_points, stream_);
  r.n_pillars = pillars_.num;
  tm.mark("voxelize");

  pillar_mean(points_.get(), pillars_, mean_.get(), stream_);
  point_features(points_.get(), r.n_points, pillars_, mean_.get(), feats_.get(), stream_);
#ifdef PCACC_WITH_TENSORRT
  pfn_->run(feats_.get(), r.n_points, pillars_, pillar_feats_.get(), stream_);
  scatter_bev(pillar_feats_.get(), mean_.get(), pillars_, bev_.get(), occ_.get(), mean_map_.get(), stream_);
#endif
  tm.mark("pillar_enc");

#ifdef PCACC_WITH_TENSORRT
  backbone_->run(bev_.get(), bev_feats_.get(), fb_seg_.get(), geo_feats_.get(), stream_);
  tm.mark("backbone");
#endif

  r.times = tm.read();
  return r;
}

}  // namespace pcacc
