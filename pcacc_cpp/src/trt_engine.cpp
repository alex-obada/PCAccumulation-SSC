#include "pcacc/trt_engine.hpp"

#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "pcacc/cuda_utils.hpp"

namespace fs = std::filesystem;

namespace pcacc {

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity sev, const char* msg) noexcept override {
    if (sev > Severity::kWARNING) return;
    std::fprintf(stderr, "[TensorRT %s] %s\n", sev <= Severity::kERROR ? "error" : "warning", msg);
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

std::string dims_str(const nvinfer1::Dims& d) {
  std::ostringstream s;
  s << '[';
  for (int i = 0; i < d.nbDims; ++i) s << (i ? ", " : "") << d.d[i];
  s << ']';
  return s.str();
}

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("TensorRT: " + what); }

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

}  // namespace

const char* precision_name(Precision p) {
  switch (p) {
    case Precision::FP32: return "fp32";
    case Precision::TF32: return "tf32";
    case Precision::FP16: return "fp16";
  }
  return "?";
}

nvinfer1::Dims dims(std::initializer_list<int64_t> d) {
  nvinfer1::Dims out{};
  if (d.size() > size_t(nvinfer1::Dims::MAX_DIMS)) fail("too many dims");
  out.nbDims = int(d.size());
  int i = 0;
  for (int64_t v : d) out.d[i++] = v;
  return out;
}

TrtEngine::TrtEngine(const std::string& models_dir, const EngineSpec& spec) : spec_(spec) {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const std::string onnx_path = models_dir + "/" + spec.name + ".onnx";
  std::ostringstream ep;
  ep << models_dir << '/' << spec.name << '.' << precision_name(spec.precision) << ".sm" << prop.major
     << prop.minor << ".trt" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH
     << ".engine";
  engine_path_ = ep.str();

  runtime_.reset(nvinfer1::createInferRuntime(logger()));
  if (!runtime_) fail("createInferRuntime failed");

  const bool have_onnx = fs::exists(onnx_path);
  const bool cache_ok = fs::exists(engine_path_) &&
                        (!have_onnx || fs::last_write_time(engine_path_) >= fs::last_write_time(onnx_path));
  if (cache_ok) {
    auto blob = read_file(engine_path_);
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) std::fprintf(stderr, "%s: cached engine unusable, rebuilding\n", engine_path_.c_str());
  }
  if (!engine_) {
    if (!have_onnx) fail(onnx_path + " not found (run export/export_models.sh)");
    auto t0 = std::chrono::steady_clock::now();
    auto blob = build(onnx_path);
    build_s_ = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
    built_now_ = true;
    std::ofstream(engine_path_, std::ios::binary).write(blob.data(), std::streamsize(blob.size()));
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) fail("deserializing the freshly built " + spec.name + " failed");
  }
  ctx_.reset(engine_->createExecutionContext());
  if (!ctx_) fail("createExecutionContext failed for " + spec.name);
  // fixed-shape engines are ready; dynamic inputs start at their opt shape
  for (auto& r : spec_.dynamic) set_shape(r.input.c_str(), r.opt);
}

TrtEngine::~TrtEngine() = default;   // unique_ptr order: context, engine, runtime (reverse of declaration)

std::vector<char> TrtEngine::build(const std::string& onnx_path) {
  std::fprintf(stderr, "building TensorRT engine %s (%s) from %s ...\n", spec_.name.c_str(),
               precision_name(spec_.precision), onnx_path.c_str());
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
  if (!builder) fail("createInferBuilder failed");
  std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));   // explicit batch (TRT 10)
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, logger()));
  if (!parser->parseFromFile(onnx_path.c_str(), int(nvinfer1::ILogger::Severity::kWARNING))) {
    std::string msg;
    for (int i = 0; i < parser->getNbErrors(); ++i) msg += std::string("\n  ") + parser->getError(i)->desc();
    fail("parsing " + onnx_path + " failed:" + msg);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, spec_.workspace_bytes);
  if (spec_.precision == Precision::FP32) cfg->clearFlag(nvinfer1::BuilderFlag::kTF32);
  if (spec_.precision == Precision::FP16) cfg->setFlag(nvinfer1::BuilderFlag::kFP16);

  // one optimization profile with every dynamic input; inputs not listed must be fixed-shape
  nvinfer1::IOptimizationProfile* prof = builder->createOptimizationProfile();
  bool any_dynamic = false;
  for (int i = 0; i < net->getNbInputs(); ++i) {
    nvinfer1::ITensor* in = net->getInput(i);
    nvinfer1::Dims d = in->getDimensions();
    bool dynamic = false;
    for (int k = 0; k < d.nbDims; ++k) dynamic |= d.d[k] < 0;
    const EngineSpec::Range* r = nullptr;
    for (auto& x : spec_.dynamic)
      if (x.input == in->getName()) r = &x;
    if (dynamic && !r) fail(spec_.name + ": input " + in->getName() + " " + dims_str(d) + " needs a shape range");
    if (!r) continue;
    using K = nvinfer1::OptProfileSelector;
    prof->setDimensions(in->getName(), K::kMIN, r->min);
    prof->setDimensions(in->getName(), K::kOPT, r->opt);
    prof->setDimensions(in->getName(), K::kMAX, r->max);
    any_dynamic = true;
  }
  if (any_dynamic) cfg->addOptimizationProfile(prof);

  std::unique_ptr<nvinfer1::IHostMemory> blob(builder->buildSerializedNetwork(*net, *cfg));
  if (!blob) fail("building " + spec_.name + " failed (see the TensorRT errors above)");
  const char* p = static_cast<const char*>(blob->data());
  return {p, p + blob->size()};
}

void TrtEngine::bind(const char* tensor, void* device_ptr) {
  if (!ctx_->setTensorAddress(tensor, device_ptr)) fail(spec_.name + ": cannot bind tensor " + tensor);
}

void TrtEngine::set_shape(const char* input, const nvinfer1::Dims& d) {
  if (!ctx_->setInputShape(input, d)) fail(spec_.name + ": invalid shape " + dims_str(d) + " for " + input);
}

void TrtEngine::enqueue(cudaStream_t s) {
  if (!ctx_->enqueueV3(s)) fail(spec_.name + ": enqueueV3 failed");
}

nvinfer1::Dims TrtEngine::shape(const char* tensor) const { return ctx_->getTensorShape(tensor); }

}  // namespace pcacc
