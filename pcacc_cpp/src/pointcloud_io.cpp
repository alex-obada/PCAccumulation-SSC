#include "pcacc/pointcloud_io.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace pcacc {

int64_t HostTensor::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

std::string HostTensor::shape_str() const {
  std::ostringstream s;
  s << '[';
  for (size_t i = 0; i < shape.size(); ++i) s << (i ? ", " : "") << shape[i];
  s << ']';
  return s.str();
}

HostTensor read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int32_t hdr[2];
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  if (!f || (hdr[0] != 0 && hdr[0] != 1) || hdr[1] < 0 || hdr[1] > 8)
    throw std::runtime_error("bad .bin header in " + path);
  HostTensor t;
  t.dtype = static_cast<DType>(hdr[0]);
  std::vector<int32_t> dims(hdr[1]);
  f.read(reinterpret_cast<char*>(dims.data()), dims.size() * sizeof(int32_t));
  t.shape.assign(dims.begin(), dims.end());
  t.data.resize(static_cast<size_t>(t.numel()) * 4);
  f.read(reinterpret_cast<char*>(t.data.data()), t.data.size());
  if (!f) throw std::runtime_error("truncated .bin file " + path);
  return t;
}

HostTensor read_bin(const std::string& path, DType dtype, const std::vector<int64_t>& shape) {
  HostTensor t = read_bin(path);
  bool ok = t.dtype == dtype && (shape.empty() || shape.size() == t.shape.size());
  for (size_t i = 0; ok && i < shape.size(); ++i) ok = shape[i] < 0 || shape[i] == t.shape[i];
  if (!ok) {
    HostTensor want;
    want.shape = shape;
    throw std::runtime_error(path + ": got dtype " + std::to_string(int(t.dtype)) + " shape " + t.shape_str() +
                             ", expected dtype " + std::to_string(int(dtype)) + " shape " + want.shape_str());
  }
  return t;
}

void write_bin(const std::string& path, DType dtype, const std::vector<int64_t>& shape, const void* data) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  int64_t n = 1;
  std::vector<int32_t> hdr{static_cast<int32_t>(dtype), static_cast<int32_t>(shape.size())};
  for (auto d : shape) {
    hdr.push_back(static_cast<int32_t>(d));
    n *= d;
  }
  f.write(reinterpret_cast<const char*>(hdr.data()), hdr.size() * sizeof(int32_t));
  f.write(reinterpret_cast<const char*>(data), n * 4);
}

Frame load_frame(const std::string& dir) {
  HostTensor pts = read_bin(dir + "/raw_points.bin", DType::F32, {-1, 3});
  HostTensor tim = read_bin(dir + "/raw_time.bin", DType::I32, {pts.dim(0)});
  Frame fr;
  fr.name = fs::path(dir).filename().string();
  const int64_t n = pts.dim(0);
  fr.xyzt.resize(n * 4);
  const float* p = pts.as<float>();
  const int32_t* t = tim.as<int32_t>();
  for (int64_t i = 0; i < n; ++i) {
    fr.xyzt[4 * i + 0] = p[3 * i + 0];
    fr.xyzt[4 * i + 1] = p[3 * i + 1];
    fr.xyzt[4 * i + 2] = p[3 * i + 2];
    fr.xyzt[4 * i + 3] = static_cast<float>(t[i]);
  }
  if (fs::exists(dir + "/poses_gt.bin")) {
    HostTensor g = read_bin(dir + "/poses_gt.bin", DType::F32, {5, 4, 4});
    fr.poses_gt.assign(g.as<float>(), g.as<float>() + g.numel());
  }
  return fr;
}

std::vector<std::string> list_frames(const std::string& data_dir) {
  std::vector<std::string> out;
  std::ifstream fl(data_dir + "/frames.txt");
  std::string line;
  while (std::getline(fl, line))
    if (!line.empty()) out.push_back(data_dir + "/" + line);
  if (out.empty() && fs::is_directory(data_dir)) {
    for (auto& e : fs::directory_iterator(data_dir))
      if (fs::exists(e.path() / "raw_points.bin")) out.push_back(e.path().string());
    std::sort(out.begin(), out.end());
  }
  return out;
}

}  // namespace pcacc
