#!/bin/bash

set -e

g++ raytracer_cpu_fresnel_reflection.cpp -o a.out
./a.out
# open image.ppm