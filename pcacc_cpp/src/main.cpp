// pcacc: reads the selected frames, runs the pipeline, prints per-stage timings.
//
//   pcacc [--data <dir>] [--models <dir>] [--precision fp32|tf32|fp16] [--frames <k>] [--dump <dir>]
//     --data      frame directories from export/dump_reference.py (default: data)
//     --models    .onnx folder; built engines are cached there (default: models)
//     --precision TensorRT precision (default fp32 = strict fp32, TF32 off)
//     --frames  run only the first k frames
//     --dump    write the intermediate outputs (points, p2v, pfn features) per frame
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

#include "pcacc/config.hpp"
#include "pcacc/pipeline.hpp"
#include "pcacc/pointcloud_io.hpp"

using namespace pcacc;
namespace fs = std::filesystem;

namespace {

void dump(const Pipeline& p, const FrameResult& r, const std::string& dir) {
  fs::create_directories(dir);
  const int n = r.n_points;
  std::vector<float4> pts(n);
  CUDA_CHECK(cudaMemcpy(pts.data(), p.points(), sizeof(float4) * n, cudaMemcpyDeviceToHost));
  std::vector<float> xyz(3 * size_t(n));
  std::vector<int32_t> t(n);
  for (int i = 0; i < n; ++i) {
    xyz[3 * i] = pts[i].x, xyz[3 * i + 1] = pts[i].y, xyz[3 * i + 2] = pts[i].z;
    t[i] = static_cast<int32_t>(pts[i].w);
  }
  write_bin(dir + "/points.bin", xyz, {n, 3});
  write_bin(dir + "/time_indice.bin", t, {n});
  std::vector<int32_t> p2v(n);
  CUDA_CHECK(cudaMemcpy(p2v.data(), p.pillars().p2v, sizeof(int) * n, cudaMemcpyDeviceToHost));
  write_bin(dir + "/p2v.bin", p2v, {n});
  std::vector<float> f(size_t(n) * kPfnIn);
  CUDA_CHECK(cudaMemcpy(f.data(), p.pfn_features(), sizeof(float) * f.size(), cudaMemcpyDeviceToHost));
  write_bin(dir + "/pfn_a_in.bin", f, {n, kPfnIn});
}

}  // namespace

int main(int argc, char** argv) {
  std::string data = "data", models = "models", precision = "fp32", dump_dir;
  int max_frames = -1;
  for (int i = 1; i < argc; ++i) {
    auto arg = [&](const char* name) { return !std::strcmp(argv[i], name) && i + 1 < argc; };
    if (arg("--data")) data = argv[++i];
    else if (arg("--models")) models = argv[++i];
    else if (arg("--precision")) precision = argv[++i];
    else if (arg("--frames")) max_frames = std::atoi(argv[++i]);
    else if (arg("--dump")) dump_dir = argv[++i];
    else {
      std::fprintf(stderr, "usage: %s [--data dir] [--models dir] [--precision fp32|tf32|fp16] [--frames k] [--dump dir]\n",
                   argv[0]);
      return 2;
    }
  }

  auto dirs = list_frames(data);
  if (max_frames >= 0 && size_t(max_frames) < dirs.size()) dirs.resize(max_frames);
  if (dirs.empty()) {
    std::fprintf(stderr, "no frames in %s (run export/dump_reference.sh)\n", data.c_str());
    return 1;
  }
  std::vector<Frame> frames;
  int max_points = 0;
  for (auto& d : dirs) {
    frames.push_back(load_frame(d));
    max_points = std::max<int>(max_points, int(frames.back().num_points()));
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("%zu frames, max %d raw points, GPU %s (sm_%d%d)\n", frames.size(), max_points, prop.name,
              prop.major, prop.minor);
#ifdef PCACC_WITH_TENSORRT
  Precision prec;
  if (precision == "fp32") prec = Precision::FP32;
  else if (precision == "tf32") prec = Precision::TF32;
  else if (precision == "fp16") prec = Precision::FP16;
  else {
    std::fprintf(stderr, "unknown --precision %s\n", precision.c_str());
    return 2;
  }
  std::printf("TensorRT precision %s, models %s\n", precision_name(prec), models.c_str());
  Pipeline pipe(max_points, models, prec);
#else
  std::printf("built without TensorRT: running the CUDA front end only (h2d .. pillar features)\n");
  Pipeline pipe(max_points);
#endif
  pipe.run(frames[0]);   // warm-up (lazy CUDA init, kernel loading)
  for (auto& f : frames) {
    FrameResult r = pipe.run(f);
    std::printf("%s  raw %7d  N %7d  M %6d |", f.name.c_str(), r.n_raw, r.n_points, r.n_pillars);
    for (auto& t : r.times) std::printf(" %s %.2f", t.stage.c_str(), t.ms);
    std::printf(" | total %.2f ms\n", r.e2e_ms());
    if (!dump_dir.empty()) dump(pipe, r, dump_dir + "/" + f.name);
  }
  return 0;
}
