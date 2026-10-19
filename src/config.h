// Shared settings for the CPU and GPU ray tracers.
// Pick a scene at compile time: -DSCENE=1 (matte), 2 (default), 3 (stress)
#pragma once

#ifndef SCENE
#define SCENE 2
#endif

#if SCENE == 1          // A: matte, no reflections (correctness baseline)
  #define FLOOR_K     0.0f
  #define BALL_K      0.0f
  #define NUM_BOUNCES 0
#elif SCENE == 2        // B: default benchmark
  #define FLOOR_K     0.1f
  #define BALL_K      0.5f
  #define NUM_BOUNCES 5
#elif SCENE == 3        // C: stress, perfect-mirror balls (needs the loop version on GPU)
  #define FLOOR_K     0.5f
  #define BALL_K      1.0f
  #define NUM_BOUNCES 10
#else
  #error "SCENE must be 1, 2 or 3"
#endif

#ifndef WIDTH
#define WIDTH  800
#endif
#ifndef HEIGHT
#define HEIGHT 600
#endif