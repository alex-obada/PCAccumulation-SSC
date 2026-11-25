#pragma once
// .bin tensor files shared with the Python tools (common/binio.py), little endian:
//   int32 dtype (0 = float32, 1 = int32), int32 ndim, int32 dims[ndim], data in C order.
#include <cstdint>
#include <string>
#include <vector>

namespace pcacc {

enum class DType : int32_t { F32 = 0, I32 = 1 };

struct HostTensor {
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;

  int64_t numel() const;
  int64_t dim(int i) const { return shape.at(i); }
  template <typename T> T* as() { return reinterpret_cast<T*>(data.data()); }
  template <typename T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
  std::string shape_str() const;
};

HostTensor read_bin(const std::string& path);
// Checks the dtype and (when given, -1 = any) the shape; throws std::runtime_error otherwise.
HostTensor read_bin(const std::string& path, DType dtype, const std::vector<int64_t>& shape = {});

void write_bin(const std::string& path, DType dtype, const std::vector<int64_t>& shape, const void* data);
inline void write_bin(const std::string& path, const std::vector<float>& v, const std::vector<int64_t>& shape) {
  write_bin(path, DType::F32, shape, v.data());
}
inline void write_bin(const std::string& path, const std::vector<int32_t>& v, const std::vector<int64_t>& shape) {
  write_bin(path, DType::I32, shape, v.data());
}

// One input frame (export/dump_reference.py): all points of the 5 sweeps, before preprocessing.
struct Frame {
  std::string name;               // "<seq>_<frame>", e.g. "000_0005"
  std::vector<float> xyzt;        // [Nr, 4]: x, y, z, sweep index (as float)
  std::vector<float> poses_gt;    // [5, 4, 4] (empty if missing)
  int64_t num_points() const { return static_cast<int64_t>(xyzt.size() / 4); }
};

Frame load_frame(const std::string& dir);
// Frame directories listed in <data_dir>/frames.txt (or every subdirectory with raw_points.bin).
std::vector<std::string> list_frames(const std::string& data_dir);

}  // namespace pcacc
