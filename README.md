# CUDA Ray Tracing — Course Project

A GPU programming project: start from a small CPU ray tracer, port it to CUDA,
add reflections, and make it fast. See [`docs/PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md)
for the full brief.

## Contents
- `starter_raytracer_cpu.cpp` — the CPU starter code (spheres on a checkerboard floor,
  diffuse shading, hard shadows). Your starting point.
- `PROBLEM_STATEMENT.md` — the project description and grading.

## Build & Run
```bash
g++ -O2 -o raytracer starter_raytracer.cpp
./raytracer            # writes image.ppm
```

## Your task
Port the pixel loop to a CUDA kernel (one thread per pixel), add reflections,
and study performance vs. image resolution and bounce depth. Details in the
problem statement.
