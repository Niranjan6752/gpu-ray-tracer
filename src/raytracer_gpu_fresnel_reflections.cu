// CUDA port of the CPU ray tracer. One thread per pixel.
// build: nvcc -O2 -arch=sm_75 -o raytracer_cuda raytracer_cuda.cu
// run:   mkdir -p ../images/output && ./raytracer_cuda   (writes ../images/output/image.ppm)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <config.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t e = (call);                                            \
        if (e != cudaSuccess) {                                            \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(e));                                \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

// #define ball_reflection 0.1f
// #define scene_reflection 0.3f
// #define NUM_BOUNCES 5

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
    float reflectivity;
    bool  isFloor;   // if true, we draw a checkerboard instead of a solid color
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

// origin is the camera for the first ray, and the bounce point on a mirror after that.
__device__ Vec3 rayColor(Vec3 origin, Vec3 dir, int bounces) {
    Vec3  lightPos(4, 6, 2);
    float tHit; //This stores the closest hit point
    int   hit = closestHit(origin, dir, tHit); //The sphere which is being hit by the ray

    // Base case 1: miss -> sky gradient (white near horizon -> blue up top).
    if (hit < 0) {
        float t = 0.5f * (dir.y + 1.0f);
        return Vec3(1, 1, 1) * (1 - t) + Vec3(0.4f, 0.6f, 1.0f) * t; // blend white and blue
    }

    // The point we hit, and the surface normal there.
    Vec3 p = origin + dir * tHit;

    // hit is the index of the sphere that was hit. 
    // p − center is an arrow from the sphere’s centre out to the hit point, 
    // which is perpendicular to the surface there.
    Vec3 n = normalize(p - scene[hit].center); // Surface Normal

    // Base color: checkerboard for the floor, solid color otherwise.
    Vec3 base = scene[hit].color;

    //CheckerBoard color Logic
    if (scene[hit].isFloor) {
        int cx = (int)floorf(p.x), cz = (int)floorf(p.z);
        bool white = ((cx + cz) & 1) == 0;
        base = white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f);
    }

    // Diffuse lighting: how directly does the surface face the light?
    Vec3  toLight = normalize(lightPos - p); // Lighting Direction
    float diffuse = fmaxf(0.0f, dot(n, toLight));

    // Hard shadow: shoot a ray toward the light; if it hits something, we're shadowed.
    float shadowT;
    int   blocked = closestHit(p + n * 0.001f, toLight, shadowT);
    float shade = (blocked >= 0) ? 0.3f : 1.0f;

    float brightness = 0.15f + diffuse * shade;   // 0.15 = a little ambient light

    float k = scene[hit].reflectivity;

    Vec3 local_color = base * brightness;

    // Base case 2: out of bounces, or a matte surface -> no reflection.
    if (bounces == 0 || k == 0) return local_color;

    Vec3 reflectedDir = dir - n * (2.0f * dot(dir, n)); // For reflection

    // Schlick's approximation for Fresnel: F = k + (1-k)(1-cos)^5
    float cosTheta = fmaxf(0.0f, dot(n, dir * -1.0f));
    float x  = 1.0f - cosTheta;
    float x2 = x * x;
    float F  = k + (1.0f - k) * (x2 * x2 * x);   // x^5 without pow (pow would run in double)

    Vec3 reflected_color = rayColor(p + n * 0.001f, reflectedDir, bounces - 1);

    return local_color * (1.0f - F) + reflected_color * F;
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

    Vec3 color = rayColor(cameraPos, dir, NUM_BOUNCES);

    int idx = (py * W + px) * 3;
    image[idx + 0] = (unsigned char)(fminf(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
}

int main() {
    const int W = WIDTH, H = HEIGHT;
    std::vector<unsigned char> image(W * H * 3);

    Sphere hostScene[NUM_SPHERES] = {
        { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), FLOOR_K, true  }, // floor
        { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), BALL_K,  false }, // red
        { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), BALL_K,  false }, // green
        { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), BALL_K,  false }, // blue
    };
    CUDA_CHECK(cudaMemcpyToSymbol(scene, hostScene, sizeof(hostScene)));

    unsigned char* d_image;
    CUDA_CHECK(cudaMalloc(&d_image, image.size()));

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // warm-up launch (not timed): the first launch pays one-time setup costs
    render<<<grid, block>>>(d_image, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, mid, stop;
    cudaEventCreate(&start); cudaEventCreate(&mid); cudaEventCreate(&stop);

    cudaEventRecord(start);
    render<<<grid, block>>>(d_image, W, H);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(mid);
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, image.size(), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0, totalMs = 0;
    cudaEventElapsedTime(&kernelMs, start, mid);
    cudaEventElapsedTime(&totalMs, start, stop);
    printf("kernel: %.3f ms   total (incl. copy back): %.3f ms\n", kernelMs, totalMs);

    cudaFree(d_image);

    const char* outPath = "../images/output/gpu.ppm";
    FILE* f = fopen(outPath, "wb");
    if (!f) {
        fprintf(stderr, "Could not open %s (does the folder exist? run: mkdir -p ../images/output)\n", outPath);
        return 1;
    }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image.data(), 1, image.size(), f);
    fclose(f);
    printf("Wrote %s (%dx%d)\n", outPath, W, H);
    return 0;
}