# pcacc_cpp

C++/CUDA implementation of MotionNet (no Python at runtime). Plan and module map: `../CLAUDE.md`.

## Inputs

```bash
bash export/dump_reference.sh                    # all selected frames -> pcacc_cpp/data/<seq>_<frame>/
bash export/dump_reference.sh --only-ref --ref 0 1 2 100 200   # only frames with reference outputs
bash export/export_models.sh                     # pcacc_cpp/models/*.onnx
```

Each frame dir has `raw_points.bin`, `raw_time.bin`, `poses_gt.bin`; frames given with `--ref` also
get `ref/` with the original model's intermediate outputs (used by the tests).

## Build

Needs `cmake`, `libeigen3-dev`, CUDA and TensorRT 10.3 (laptop: tar archive in `~/TensorRT-10.3`;
Jetson: part of JetPack). Presets are in `CMakePresets.json`:

```bash
cmake --preset laptop && cmake --build --preset laptop      # sm_89, gcc-13, TENSORRT_ROOT=~/TensorRT-10.3
cmake --preset jetson && cmake --build --preset jetson      # sm_87, TensorRT from /usr
```

Without TensorRT the project still builds (CUDA modules + tests) and the engine stages are skipped.
In VS Code, CMake Tools picks the presets up and IntelliSense uses `build/compile_commands.json`.

## Run

```bash
./build/pcacc --data data --models models [--precision fp32|tf32|fp16] [--frames k] [--dump out_dir]
ctest --preset laptop                            # or: ./build/test_modules all data
```

## Status

| Module | File | Test vs. original |
|---|---|---|
| crop + ground removal | `src/cuda/preprocess.cu` | exact |
| voxelisation | `src/cuda/voxelize.cu` | same pillar set and point assignment (numbering sorted by key) |
| pillar mean, PFN features | `src/cuda/pillar_ops.cu` | ≤ 1.2e-6 abs |
| pillar max + concat, BEV scatter | `src/cuda/pillar_ops.cu` | exact |
| TensorRT engines (build / cache / run) | `src/trt_engine.cpp` | all 6 `.onnx` build in TRT 10.3 |
| pillar encoder (pfn_a/b/c + max/concat kernels) | `src/networks.cpp` | pfn ≤ 1.4e-6 abs, pillar features ≤ 3.6e-7 |
| backbone (UNet + heads) | `src/networks.cpp` | `fb_seg` ~1e-4 abs, argmax 100 % identical |
| ego-motion, motion segmentation, DBSCAN | | todo |
