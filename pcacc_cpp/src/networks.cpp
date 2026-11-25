#include "pcacc/networks.hpp"

#include <stdexcept>

namespace pcacc {

namespace {

EngineSpec pfn_spec(const char* name, int in_ch, Precision p) {
  EngineSpec s;
  s.name = name;
  s.precision = p;
  s.dynamic = {{"features", dims({1, in_ch}), dims({kPfnOptPoints, in_ch}), dims({kMaxPoints, in_ch})}};
  return s;
}

}  // namespace

PillarEncoder::PillarEncoder(const std::string& models_dir, Precision p)
    : a(models_dir, pfn_spec("pfn_a", kPfnIn, p)),
      b(models_dir, pfn_spec("pfn_b", 2 * kC, p)),
      c(models_dir, pfn_spec("pfn_c", 2 * kC, p)),
      net_(size_t(kMaxPoints) * kC),
      cat_(size_t(kMaxPoints) * 2 * kC),
      pmax_(size_t(kMaxPoints) * kC) {}

void PillarEncoder::run_engine(TrtEngine& e, const float* in, int n, float* out, cudaStream_t s) {
  if (n < 1 || n > kMaxPoints) throw std::runtime_error(e.name() + ": point count outside the engine profile");
  const bool first = e.name() == "pfn_a";
  e.set_shape("features", dims({n, first ? kPfnIn : 2 * kC}));
  e.bind("features", const_cast<float*>(in));
  e.bind(e.name() == "pfn_c" ? "feats" : "net", out);
  e.enqueue(s);
}

void PillarEncoder::run(const float* features, int n, const Pillars& p, float* pillar_feats, cudaStream_t s) {
  run_engine(a, features, n, net_.get(), s);
  pillar_max(net_.get(), kC, p, pmax_.get(), s);
  gather_concat(net_.get(), pmax_.get(), kC, n, p, cat_.get(), s);
  run_engine(b, cat_.get(), n, net_.get(), s);
  pillar_max(net_.get(), kC, p, pmax_.get(), s);
  gather_concat(net_.get(), pmax_.get(), kC, n, p, cat_.get(), s);
  run_engine(c, cat_.get(), n, net_.get(), s);
  pillar_max(net_.get(), kC, p, pillar_feats, s);
}

Backbone::Backbone(const std::string& models_dir, Precision p) : engine(models_dir, EngineSpec{"backbone", p, {}}) {}

void Backbone::run(const float* bev, float* bev_feats, float* fb_seg, float* geo_feats, cudaStream_t s) {
  engine.bind("bev", const_cast<float*>(bev));
  engine.bind("bev_feats", bev_feats);
  engine.bind("fb_seg", fb_seg);
  engine.bind("geo_feats", geo_feats);
  engine.enqueue(s);
}

}  // namespace pcacc
