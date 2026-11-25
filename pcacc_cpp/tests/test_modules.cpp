// Module tests: each one runs a single C++ module on the exact reference input of the original
// model (data/<frame>/ref/, written by export/dump_reference.py) and compares the output.
//
//   test_modules <test> <data_dir> [models_dir]
//     test = preprocess | voxelize | pillar_mean | point_features | pillar_max_concat | bev
//            | pfn_engines | pillar_encoder | backbone (TensorRT builds only) | all
//     models_dir (default: <data_dir>/../models) holds the .onnx files; engines are cached there.
// Exit code 0 = pass on every frame with a ref/ directory.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "pcacc/config.hpp"
#include "pcacc/cuda_utils.hpp"
#include "pcacc/modules.hpp"
#include "pcacc/pointcloud_io.hpp"
#ifdef PCACC_WITH_TENSORRT
#include "pcacc/networks.hpp"
#endif

namespace fs = std::filesystem;
using namespace pcacc;

namespace {

// ---------------------------------------------------------------- comparison helpers
struct Check {
  bool ok = true;
  void fail(const char* fmt, const std::string& what, double a = 0, double b = 0) {
    ok = false;
    std::printf("    FAIL %s: ", what.c_str());
    std::printf(fmt, a, b);
    std::printf("\n");
  }
};

// |a - b| <= atol + rtol * |b|, element-wise; prints the worst error
void expect_close(Check& ck, const std::string& what, const float* a, const float* b, size_t n,
                  double rtol = 1e-4, double atol = 1e-5) {
  double max_abs = 0, max_ref = 0;
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    double d = std::fabs(double(a[i]) - b[i]);
    max_abs = std::max(max_abs, d);
    max_ref = std::max(max_ref, std::fabs(double(b[i])));
    if (!(d <= atol + rtol * std::fabs(b[i]))) ++bad;
  }
  std::printf("    %-22s n=%-9zu max|diff| %.3g (max|ref| %.3g)  out of tol: %zu\n", what.c_str(), n, max_abs,
              max_ref, bad);
  if (bad) ck.fail("%g elements out of tolerance", what, double(bad));
}

// network outputs vs PyTorch (fp32): |a - b| <= 1e-4 * |b| + 1e-4 * max|b|
void expect_close_net(Check& ck, const std::string& what, const float* a, const float* b, size_t n) {
  double max_ref = 0;
  for (size_t i = 0; i < n; ++i) max_ref = std::max(max_ref, std::fabs(double(b[i])));
  expect_close(ck, what, a, b, n, 1e-4, 1e-4 * max_ref);
}

void expect_eq(Check& ck, const std::string& what, long long a, long long b) {
  if (a != b) ck.fail("%g != %g", what, double(a), double(b));
}

int key_of(int y, int x, int t) { return (t * kNy + y) * kNx + x; }

// ---------------------------------------------------------------- per-frame fixture
struct Ref {
  std::string dir;
  HostTensor t(const std::string& name) const { return read_bin(dir + "/ref/" + name + ".bin"); }
};

struct Gpu {
  cudaStream_t s{};
  Gpu() { CUDA_CHECK(cudaStreamCreate(&s)); }
  ~Gpu() { cudaStreamDestroy(s); }
};

// reference points [N, 3] + time [N] -> float4 points on the GPU
DeviceBuffer<float4> upload_ref_points(const Ref& r, int& n, cudaStream_t s) {
  HostTensor p = read_bin(r.dir + "/ref/points.bin", DType::F32, {-1, 3});
  HostTensor t = read_bin(r.dir + "/ref/time_indice.bin", DType::I32, {p.dim(0)});
  n = static_cast<int>(p.dim(0));
  std::vector<float4> h(n);
  for (int i = 0; i < n; ++i)
    h[i] = make_float4(p.as<float>()[3 * i], p.as<float>()[3 * i + 1], p.as<float>()[3 * i + 2],
                       float(t.as<int32_t>()[i]));
  DeviceBuffer<float4> d(std::max(n, 1));
  d.upload(h.data(), n, s);
  return d;
}

template <typename T>
DeviceBuffer<T> upload(const HostTensor& h, cudaStream_t s) {
  DeviceBuffer<T> d(std::max<int64_t>(h.numel(), 1));
  d.upload(h.as<T>(), h.numel(), s);
  return d;
}

// C++ pillar index for each reference pillar (matched by the (t, y, x) key)
std::vector<int> match_pillars(const HostTensor& ref_coords, const std::vector<int4>& cpp_coords) {
  std::map<int, int> cpp_of_key;
  for (size_t m = 0; m < cpp_coords.size(); ++m)
    cpp_of_key[key_of(cpp_coords[m].y, cpp_coords[m].z, cpp_coords[m].w)] = int(m);
  std::vector<int> out(ref_coords.dim(0), -1);
  const int32_t* c = ref_coords.as<int32_t>();
  for (int64_t r = 0; r < ref_coords.dim(0); ++r) {
    auto it = cpp_of_key.find(key_of(c[4 * r + 1], c[4 * r + 2], c[4 * r + 3]));
    if (it != cpp_of_key.end()) out[r] = it->second;
  }
  return out;
}

// reorder a per-pillar C++ tensor [M, C] into the reference numbering
std::vector<float> to_ref_order(const std::vector<float>& cpp, int ch, const std::vector<int>& match) {
  std::vector<float> out(match.size() * ch, NAN);
  for (size_t r = 0; r < match.size(); ++r)
    if (match[r] >= 0) std::copy_n(&cpp[size_t(match[r]) * ch], ch, &out[r * ch]);
  return out;
}

std::vector<int4> download_coords(const Pillars& p, cudaStream_t s) {
  std::vector<int4> h(p.num);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), p.coords, sizeof(int4) * p.num, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

template <typename T>
std::vector<T> download(const T* d, size_t n, cudaStream_t s) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), d, sizeof(T) * n, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return h;
}

// ---------------------------------------------------------------- tests
void test_preprocess(const Ref& r, Check& ck) {
  Gpu g;
  Frame f = load_frame(r.dir);
  const int nr = int(f.num_points());
  DeviceBuffer<float4> raw(nr), out(nr);
  raw.upload(reinterpret_cast<const float4*>(f.xyzt.data()), nr, g.s);
  Preprocessor pre(nr);
  int n = pre.run(raw.get(), nr, out.get(), g.s);

  HostTensor rp = r.t("points"), rt = r.t("time_indice");
  expect_eq(ck, "N", n, rp.dim(0));
  std::printf("    raw %d -> N %d (ref %lld)\n", nr, n, (long long)rp.dim(0));
  if (n != rp.dim(0)) return;
  auto h = out.download(n, g.s);
  long long diff = 0;
  for (int i = 0; i < n; ++i) {
    const float* q = rp.as<float>() + 3 * i;
    diff += h[i].x != q[0] || h[i].y != q[1] || h[i].z != q[2] || int(h[i].w) != rt.as<int32_t>()[i];
  }
  expect_eq(ck, "points differing", diff, 0);
}

void test_voxelize(const Ref& r, Check& ck) {
  Gpu g;
  int n;
  auto pts = upload_ref_points(r, n, g.s);
  Voxelizer vox(n);
  Pillars p = vox.run(pts.get(), n, g.s);

  HostTensor rc = r.t("coordinates"), rp2v = r.t("p2v"), rnum = r.t("num_points");
  std::printf("    N %d -> M %d (ref %lld)\n", n, p.num, (long long)rc.dim(0));
  expect_eq(ck, "M", p.num, rc.dim(0));
  auto coords = download_coords(p, g.s);
  auto p2v = download(p.p2v, n, g.s);
  auto start = download(p.start, p.num + 1, g.s);

  // pillar set, as sorted keys
  std::vector<int> kc, kr;
  for (auto& c : coords) kc.push_back(key_of(c.y, c.z, c.w));
  const int32_t* c = rc.as<int32_t>();
  for (int64_t m = 0; m < rc.dim(0); ++m) kr.push_back(key_of(c[4 * m + 1], c[4 * m + 2], c[4 * m + 3]));
  std::sort(kr.begin(), kr.end());
  expect_eq(ck, "pillar keys sorted", std::is_sorted(kc.begin(), kc.end()), 1);
  expect_eq(ck, "pillar sets differ", kc != kr, 0);

  // same pillar for every point, same number of points per pillar
  long long wrong = 0, wrong_cnt = 0;
  for (int i = 0; i < n; ++i) {
    int pr = rp2v.as<int32_t>()[i], pc = p2v[i];
    wrong += pc < 0 || key_of(coords[pc].y, coords[pc].z, coords[pc].w) !=
                           key_of(c[4 * pr + 1], c[4 * pr + 2], c[4 * pr + 3]);
  }
  auto match = match_pillars(rc, coords);
  for (int64_t m = 0; m < rc.dim(0); ++m)
    wrong_cnt += match[m] < 0 || start[match[m] + 1] - start[match[m]] != rnum.as<int32_t>()[m];
  expect_eq(ck, "points in wrong pillar", wrong, 0);
  expect_eq(ck, "pillars with wrong count", wrong_cnt, 0);
}

// shared setup for the pillar_ops tests: reference points voxelized by the C++ module
struct PillarFixture {
  Gpu g;
  int n = 0;
  DeviceBuffer<float4> pts;
  Voxelizer vox;
  Pillars p;
  std::vector<int> match;
  HostTensor ref_coords;
  explicit PillarFixture(const Ref& r)
      : pts(upload_ref_points(r, n, g.s)), vox(n), p(vox.run(pts.get(), n, g.s)), ref_coords(r.t("coordinates")) {
    match = match_pillars(ref_coords, download_coords(p, g.s));
  }
};

void test_pillar_mean(const Ref& r, Check& ck) {
  PillarFixture fx(r);
  DeviceBuffer<float> mean(fx.p.num * 3);
  pillar_mean(fx.pts.get(), fx.p, mean.get(), fx.g.s);
  auto got = to_ref_order(mean.download(fx.p.num * 3, fx.g.s), 3, fx.match);
  HostTensor ref = r.t("pillar_mean");
  expect_close(ck, "pillar_mean", got.data(), ref.as<float>(), ref.numel());
}

void test_point_features(const Ref& r, Check& ck) {
  PillarFixture fx(r);
  DeviceBuffer<float> mean(fx.p.num * 3), feats(size_t(fx.n) * kPfnIn);
  pillar_mean(fx.pts.get(), fx.p, mean.get(), fx.g.s);
  point_features(fx.pts.get(), fx.n, fx.p, mean.get(), feats.get(), fx.g.s);
  auto got = feats.download(feats.size(), fx.g.s);
  HostTensor ref = r.t("pfn_a_in");
  expect_close(ck, "pfn_a_in", got.data(), ref.as<float>(), ref.numel());
}

void test_pillar_max_concat(const Ref& r, Check& ck) {
  PillarFixture fx(r);
  const int n = fx.n, m = fx.p.num;
  DeviceBuffer<float> pmax(size_t(m) * kC), cat(size_t(n) * 2 * kC);
  for (auto [in, out] : {std::pair{"pfn_a_out", "pfn_b_in"}, std::pair{"pfn_b_out", "pfn_c_in"}}) {
    auto net = upload<float>(r.t(in), fx.g.s);
    pillar_max(net.get(), kC, fx.p, pmax.get(), fx.g.s);
    gather_concat(net.get(), pmax.get(), kC, n, fx.p, cat.get(), fx.g.s);
    HostTensor ref = r.t(out);
    auto got = cat.download(cat.size(), fx.g.s);
    expect_close(ck, out, got.data(), ref.as<float>(), ref.numel(), 0, 0);   // pure data movement: exact
  }
  auto net = upload<float>(r.t("pfn_c_out"), fx.g.s);
  pillar_max(net.get(), kC, fx.p, pmax.get(), fx.g.s);
  auto got = to_ref_order(pmax.download(pmax.size(), fx.g.s), kC, fx.match);
  HostTensor ref = r.t("pillar_feats");
  expect_close(ck, "pillar_feats", got.data(), ref.as<float>(), ref.numel(), 0, 0);
}

void test_bev(const Ref& r, Check& ck) {
  PillarFixture fx(r);
  const int m = fx.p.num;
  constexpr size_t hw = size_t(kNy) * kNx;
  DeviceBuffer<float> pmax(size_t(m) * kC), mean(size_t(m) * 3), bev(kT * kC * hw), occ(kT * hw),
      mean_map(kT * 3 * hw);
  auto net = upload<float>(r.t("pfn_c_out"), fx.g.s);
  pillar_max(net.get(), kC, fx.p, pmax.get(), fx.g.s);
  pillar_mean(fx.pts.get(), fx.p, mean.get(), fx.g.s);
  scatter_bev(pmax.get(), mean.get(), fx.p, bev.get(), occ.get(), mean_map.get(), fx.g.s);
  HostTensor ref = r.t("bev");
  auto got = bev.download(bev.size(), fx.g.s);
  expect_close(ck, "bev", got.data(), ref.as<float>(), ref.numel(), 0, 0);
  auto o = occ.download(occ.size(), fx.g.s);
  long long n_occ = 0;
  for (float v : o) n_occ += v == 1.f;
  expect_eq(ck, "occupied cells != M", n_occ, m);
}

#ifdef PCACC_WITH_TENSORRT
// ---------------------------------------------------------------- TensorRT modules (strict fp32)
std::string g_models;

PillarEncoder& encoder() {
  static PillarEncoder e(g_models, Precision::FP32);
  return e;
}

Backbone& backbone() {
  static Backbone b(g_models, Precision::FP32);
  return b;
}

// each engine on its exact reference input
void test_pfn_engines(const Ref& r, Check& ck) {
  Gpu g;
  PillarEncoder& enc = encoder();
  for (auto [e, name] : {std::pair{&enc.a, "pfn_a"}, std::pair{&enc.b, "pfn_b"}, std::pair{&enc.c, "pfn_c"}}) {
    HostTensor in = r.t(std::string(name) + "_in"), ref = r.t(std::string(name) + "_out");
    auto d_in = upload<float>(in, g.s);
    DeviceBuffer<float> out(ref.numel());
    PillarEncoder::run_engine(*e, d_in.get(), int(in.dim(0)), out.get(), g.s);
    auto got = out.download(out.size(), g.s);
    expect_close_net(ck, std::string(name) + "_out", got.data(), ref.as<float>(), ref.numel());
  }
}

// pfn_a_in -> pfn_a/b/c with the C++ pillar max / concat kernels in between -> pillar features
void test_pillar_encoder(const Ref& r, Check& ck) {
  PillarFixture fx(r);
  auto feats = upload<float>(r.t("pfn_a_in"), fx.g.s);
  DeviceBuffer<float> pf(size_t(fx.p.num) * kC);
  encoder().run(feats.get(), fx.n, fx.p, pf.get(), fx.g.s);
  auto got = to_ref_order(pf.download(pf.size(), fx.g.s), kC, fx.match);
  HostTensor ref = r.t("pillar_feats");
  expect_close_net(ck, "pillar_feats", got.data(), ref.as<float>(), ref.numel());
}

void test_backbone(const Ref& r, Check& ck) {
  Gpu g;
  constexpr size_t hw = size_t(kNy) * kNx;
  auto bev = upload<float>(r.t("bev"), g.s);
  DeviceBuffer<float> feats(kT * kC * hw), seg(kT * kSegC * hw), geo(kT * kGeoC * hw);
  backbone().run(bev.get(), feats.get(), seg.get(), geo.get(), g.s);
  HostTensor ref = r.t("fb_seg");
  auto got = seg.download(seg.size(), g.s);
  expect_close_net(ck, "fb_seg", got.data(), ref.as<float>(), ref.numel());
  // background / foreground decision per cell (what ego-motion and the point selection use)
  const float* rf = ref.as<float>();
  size_t same = 0;
  for (int t = 0; t < kT; ++t)
    for (size_t i = 0; i < hw; ++i) {
      const size_t b0 = (size_t(t) * kSegC) * hw + i, b1 = b0 + hw;
      same += (got[b1] > got[b0]) == (rf[b1] > rf[b0]);
    }
  const double agree = double(same) / (kT * hw);
  std::printf("    fb_seg argmax agreement %.5f %%\n", 100 * agree);
  if (agree < 0.9999) ck.fail("argmax agreement %.5f %% < 99.99 %%", "fb_seg", 100 * agree);
}
#endif

const std::map<std::string, std::function<void(const Ref&, Check&)>> kTests = {
    {"preprocess", test_preprocess},         {"voxelize", test_voxelize},
    {"pillar_mean", test_pillar_mean},       {"point_features", test_point_features},
    {"pillar_max_concat", test_pillar_max_concat}, {"bev", test_bev},
#ifdef PCACC_WITH_TENSORRT
    {"pfn_engines", test_pfn_engines},       {"pillar_encoder", test_pillar_encoder},
    {"backbone", test_backbone},
#endif
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: %s <test|all> <data_dir> [models_dir]\ntests:", argv[0]);
    for (auto& [k, _] : kTests) std::printf(" %s", k.c_str());
    std::printf("\n");
    return 2;
  }
  std::string which = argv[1], data = argv[2];
#ifdef PCACC_WITH_TENSORRT
  g_models = argc > 3 ? argv[3] : data + "/../models";
#endif
  std::vector<Ref> refs;
  for (auto& d : list_frames(data))
    if (fs::is_directory(d + "/ref")) refs.push_back({d});
  if (refs.empty()) {
    std::printf("no frames with ref/ in %s (run export/dump_reference.sh)\n", data.c_str());
    return 1;
  }
  bool all_ok = true;
  for (auto& [name, fn] : kTests) {
    if (which != "all" && which != name) continue;
    for (auto& r : refs) {
      std::printf("[%s] %s\n", name.c_str(), fs::path(r.dir).filename().c_str());
      Check ck;
      try {
        fn(r, ck);
      } catch (const std::exception& e) {
        std::printf("    FAIL exception: %s\n", e.what());
        ck.ok = false;
      }
      std::printf("    %s\n", ck.ok ? "ok" : "FAILED");
      all_ok &= ck.ok;
    }
  }
  return all_ok ? 0 : 1;
}
