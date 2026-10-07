// CUDA port of the CPU ray tracer. One thread per pixel.
// build: nvcc -O2 -o raytracer_cuda raytracer_cuda.cu
// run:   ./raytracer_cuda   (writes image.ppm)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(e));                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// default ctor has to stay trivial or __constant__ won't compile
struct Vec3 {
    float x, y, z;
    Vec3() = default;
    __host__ __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};
__device__ Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
__device__ float dot(Vec3 a, Vec3 b)  { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ Vec3  normalize(Vec3 a)    { float len = sqrtf(dot(a, a)); return a * (1.0f / len); }

struct Sphere {
    Vec3  center;
    float radius;
    Vec3  color;
    bool  isFloor;
};

#define NUM_SPHERES 4
__constant__ Sphere scene[NUM_SPHERES];   // filled from hostScene in main

__device__ float hitSphere(const Sphere& s, Vec3 origin, Vec3 dir) {
    Vec3  oc = origin - s.center;
    float a = dot(dir, dir);
    float b = 2.0f * dot(oc, dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0) return -1.0f;
    float t = (-b - sqrtf(disc)) / (2 * a);
    return (t > 0.001f) ? t : -1.0f;
}

__device__ int closestHit(Vec3 origin, Vec3 dir, float& tHit) {
    int   best = -1;
    float closest = 1e30f;
    for (int i = 0; i < NUM_SPHERES; i++) {
        float t = hitSphere(scene[i], origin, dir);
        if (t > 0 && t < closest) { closest = t; best = i; }
    }
    tHit = closest;
    return best;
}

__device__ Vec3 rayColor(Vec3 origin, Vec3 dir) {
    // x = right, y = up, z = toward the camera
    Vec3  lightPos(4, 6, 2);
    float tHit;
    int   hit = closestHit(origin, dir, tHit);

    if (hit < 0) {
        float t = 0.5f * (dir.y + 1.0f);
        return Vec3(1, 1, 1) * (1 - t) + Vec3(0.4f, 0.6f, 1.0f) * t;
    }

    Vec3 p = origin + dir * tHit;
    Vec3 n = normalize(p - scene[hit].center);

    Vec3 base = scene[hit].color;
    if (scene[hit].isFloor) {
        int cx = (int)floorf(p.x), cz = (int)floorf(p.z);
        bool white = ((cx + cz) & 1) == 0;
        base = white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f);
    }

    Vec3  toLight = normalize(lightPos - p);
    float diffuse = fmaxf(0.0f, dot(n, toLight));

    // shadow ray toward the light, offset a bit so it doesn't hit the surface it started on
    float shadowT;
    int   blocked = closestHit(p + n * 0.001f, toLight, shadowT);
    float shade = (blocked >= 0) ? 0.3f : 1.0f;

    float brightness = 0.15f + diffuse * shade;
    return base * brightness;
}

__global__ void render(unsigned char* image, int W, int H) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;   // grid is rounded up past the edge

    Vec3  cameraPos(0, 1, 3);
    float aspect = (float)W / H;

    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3  dir = normalize(Vec3(u, v, -1.5f));

    Vec3 color = rayColor(cameraPos, dir);

    int idx = (py * W + px) * 3;
    image[idx + 0] = (unsigned char)(fminf(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
}

int main() {
    const int W = 800, H = 600;
    std::vector<unsigned char> image(W * H * 3);

    Sphere hostScene[NUM_SPHERES] = {
        { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), true  }, // floor
        { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), false }, // red
        { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), false }, // green
        { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), false }, // blue
    };
    CUDA_CHECK(cudaMemcpyToSymbol(scene, hostScene, sizeof(hostScene)));

    unsigned char* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, image.size()));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    render<<<grid, block>>>(d_image, W, H);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("kernel: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(image.data(), d_image, image.size(), cudaMemcpyDeviceToHost));
    cudaFree(d_image);

    FILE* f = fopen("image.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image.data(), 1, image.size(), f);
    fclose(f);
    printf("Wrote image.ppm (%dx%d)\n", W, H);
    return 0;
}
