# CUDA Ray Tracing — Course Project

This repository contains the starter code and project specification for the CUDA Ray Tracing project.

## Contents
- `starter_raytracer_cpu.cpp` — the CPU starter code (spheres on a checkerboard floor,
  diffuse shading, hard shadows). Your starting point.
- `PROBLEM_STATEMENT.md` — the project description and grading.

## Build & Run
```bash
g++ -O2 -o raytracer starter_raytracer_cpu.cpp
./raytracer            # writes image.ppm
```

## Your task
Implement your own Ray Tracer. Start from a small CPU ray tracer, port it to CUDA, add reflections, and make it fast. 
