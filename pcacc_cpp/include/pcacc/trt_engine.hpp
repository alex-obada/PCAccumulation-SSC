#pragma once
// TensorRT engine wrapper: load a cached .engine or build it from the .onnx, bind GPU buffers by
// tensor name, run with enqueueV3 on a stream.
//
// Engines are built on the machine that runs them (never copied between machines). The cache file
// name holds the precision, the GPU arch and the TensorRT version:
//   <models>/<name>.<fp32|tf32|fp16>.sm<XY>.trt<A.B.C>.engine
// It is rebuilt when missing, when it fails to deserialize, or when the .onnx is newer.
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace pcacc {

// FP32 = strict fp32 (TF32 off, the reference for the accuracy tests), TF32 = TensorRT's default
// for fp32 networks (TF32 convs / matmuls, like the V0 baseline), FP16 = fp16 kernels allowed.
enum class Precision { FP32, TF32, FP16 };
const char* precision_name(Precision p);

struct EngineSpec {
  std::string name;                 // models/<name>.onnx
  Precision precision = Precision::FP32;
  struct Range {                    // optimization profile of one dynamic input
    std::string input;
    nvinfer1::Dims min, opt, max;
  };
  std::vector<Range> dynamic;
  size_t workspace_bytes = size_t(1) << 30;   // builder memory pool limit (keep low on the Jetson)
};

// nvinfer1::Dims from a list, e.g. dims({4, 1024, 3})
nvinfer1::Dims dims(std::initializer_list<int64_t> d);

class TrtEngine {
 public:
  TrtEngine(const std::string& models_dir, const EngineSpec& spec);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  // Binds a device buffer to an input or output tensor (kept until rebound).
  void bind(const char* tensor, void* device_ptr);
  // Sets the shape of a dynamic input (checked against the profile range).
  void set_shape(const char* input, const nvinfer1::Dims& d);
  // Runs asynchronously on the stream. All tensors must be bound and dynamic shapes set.
  void enqueue(cudaStream_t s);

  nvinfer1::Dims shape(const char* tensor) const;   // current shape (after set_shape for outputs)
  const std::string& name() const { return spec_.name; }
  const std::string& engine_path() const { return engine_path_; }
  bool built_now() const { return built_now_; }      // false = loaded from the cache
  float build_seconds() const { return build_s_; }

 private:
  std::vector<char> build(const std::string& onnx_path);

  EngineSpec spec_;
  std::string engine_path_;
  bool built_now_ = false;
  float build_s_ = 0.f;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
};

}  // namespace pcacc
