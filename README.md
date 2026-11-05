# CUDA Ray Tracer (CS6023 course project)

Niranjan, CS26M029, IIT Madras

A ray tracer that starts as a small CPU program, is ported to CUDA, gets reflections and a
triangle mesh (the Stanford bunny, 69,451 triangles), and is then sped up with a BVH and a
few GPU-specific changes. The problem statement is in `docs/PROBLEM_STATEMENT.md`.

## Requirements

- NVIDIA GPU and the CUDA toolkit (`nvcc`). Tested on a GTX 1650 (sm_75), CUDA 13.0, driver 581.42.
- `g++` and `make`. On Windows I used MSYS2 g++ and GNU make from winget.
- Python 3 for `compare.py`, which compares two images.

If your GPU is not Turing, pass its architecture, e.g. `make run ARCH=sm_86`.

## How to run

All commands run from the `src` folder. Images are written to `images/output/`.

```
cd src
make help
```

| Command | What it does |
|---|---|
| `make` | build every program |
| `make spheres` | step 1: spheres with reflections, CPU vs first CUDA port, image compare |
| `make run` | step 2: bunny with BVH, CPU vs GPU at 800x600, image compare |
| `make brute` | step 3: GPU brute force vs GPU BVH at 200x150 |
| `make cpubrute` | step 3b: CPU brute force vs CPU BVH at 200x150 (slow, about 2 minutes) |
| `make opt` | step 4: the GPU optimisations one by one at 1920x1080, median of 9 launches |
| `make sweep` | resolution, bounce count and scene size sweeps with the best version |
| `make heat` | heat maps of BVH nodes and triangles tested per pixel |
| `make demo` | steps 1 to 4 in one go |
| `make clean` | delete the built programs (needed after changing a setting) |

Settings can be changed on the command line, e.g. `make run SCENE=3 WIDTH=1920 HEIGHT=1080`:

- `SCENE`: 1 matte, 2 default (5 bounces), 3 mirror (10 bounces)
- `WIDTH`, `HEIGHT`: image size for `spheres` and `run`
- `BALLS`: 0 bunny only (used for all results below), 1 keeps the three spheres
- `ARCH`: GPU architecture

## Source files

| File | What it is |
|---|---|
| `starter_raytracer_cpu.cpp` | the given starter code |
| `raytracer_cpu_multi_reflection.cpp` | CPU, spheres with multiple reflections |
| `raytracer_cpu_fresnel_reflection.cpp` | CPU, spheres with Fresnel reflections |
| `raytracer_gpu_fresnel_reflections.cu` | first CUDA port of the above |
| `raytracer_cpu_mesh.cpp` | CPU, loads the bunny, tests every triangle (no BVH) |
| `raytracer_cpu_bvh.cpp` | CPU with a BVH (`-DBRUTE` turns the BVH off) |
| `raytracer_gpu_bvh.cu` | CUDA with the BVH (`-DBRUTE` turns the BVH off) |
| `raytracer_gpu_exp.cu` | CUDA experiment build: node layout, block size, run-time arguments, timing over many launches, heat maps |
| `config.h` | shared scene settings |
| `compare.py` | pixel-by-pixel comparison of two PPM images |
| `run_progression.ps1`, `run_experiments.ps1` | Windows scripts used to collect the results |
| `profiles/` | Nsight Compute reports (text exports) |

## Results (GTX 1650, measured)

| Comparison | Time |
|---|---|
| Spheres, 800x600: CPU vs GPU kernel | ~76 ms vs 0.16 ms |
| Bunny, 200x150: GPU brute force vs GPU BVH | 124 ms vs 0.34 ms |
| Bunny, 200x150: CPU brute force vs CPU BVH | 124 s vs 75 ms |
| Bunny, 800x600, scene 2: CPU BVH vs GPU BVH | 427 ms (median) vs 2.44 ms, about 175x |

GPU optimisations, 1920x1080, scene 2, median of 9 kernel launches. Every version gives the
same image (same checksum).

| Version | Kernel time |
|---|---|
| layout 0, 16x16 blocks (first version) | ~5.0 ms |
| layout 0, 8x8 blocks | ~4.7 ms |
| float4 layout, 16x16 blocks | ~3.7 ms |
| float4 layout, 8x4 blocks (occupancy capped at 50%) | ~4.1 ms |
| float4 layout, 8x8 blocks (best) | ~3.65 ms |

The float4 layout packs a BVH node into 32 bytes and a triangle into 48 bytes with precomputed
edges. In Nsight Compute it cut memory sectors from 54.8M to 16.8M. CPU and GPU images match
except for 2 of 480,000 pixels (max difference 140). They are isolated pixels inside the image,
not on the bunny's outline; I did not investigate the cause.

CPU numbers are single-threaded and were taken with the laptop on charger, apps closed, first
runs discarded.

## Use of AI

I used an AI assistant (Claude) for explanations, to write the experiment harness
(`raytracer_gpu_exp.cu`, the `.ps1` scripts, the Makefile) and for guidance on profiling with
Nsight Compute. I ran all the programs and checked all the results myself.
