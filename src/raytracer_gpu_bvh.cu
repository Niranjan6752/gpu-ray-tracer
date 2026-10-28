// CUDA port of the CPU mesh + BVH ray tracer (raytracer_cpu_bvh.cpp). One thread per pixel.
// The CPU loads the OBJ and builds the BVH; the triangles and BVH nodes are copied to GLOBAL memory
// (70k triangles are ~2.5 MB, far over the 64 KB __constant__ limit). The three balls stay in __constant__.
//
// build (BVH):    nvcc -O2 -arch=sm_75 -DSCENE=2 -DNO_BALLS -o gpu_bvh raytracer_gpu_bvh.cu
// build (brute):  nvcc -O2 -arch=sm_75 -DSCENE=2 -DNO_BALLS -DBRUTE -o gpu_brute raytracer_gpu_bvh.cu
// run:            mkdir -p ../images/output && ./gpu_bvh   (writes ../images/output/gpu_bvh.ppm)
// Needs stanford-bunny.obj and config.h in the same folder (src).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "config.h"

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
__host__ __device__ Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
__host__ __device__ float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__host__ __device__ Vec3 normalize(Vec3 a) { float len = sqrtf(dot(a, a)); return a * (1.0f / len); }
__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(a.y * b.z - b.y * a.z,
                b.x * a.z - a.x * b.z,
                a.x * b.y - b.x * a.y);
}

struct Sphere {
    Vec3  center;
    float radius;
    Vec3  color;
    float reflectivity;
    bool  isFloor;   // if true, we draw a checkerboard instead of a solid color
};

#ifdef NO_BALLS
#define NUM_SPHERES 1
#else
#define NUM_SPHERES 4
#endif
__constant__ Sphere scene[NUM_SPHERES];   // filled from hostScene in main

struct Triangle { Vec3 v0, v1, v2; };
struct BVHNode {
    Vec3 bmin, bmax;   // box around everything below this node
    int  rightOrFirst; // inner node: index of the RIGHT child (the left child is the next node)
                       // leaf: index of its first triangle
    int  count;        // 0 = inner node; >0 = leaf with this many triangles
};

// What the kernel needs to find the mesh. Pointers are device (global memory) addresses.
struct Mesh {
    const Triangle* tris;
    const BVHNode*  nodes;
    int             numTris;
};

struct Hit { float t; Vec3 n; Vec3 color; float k; bool isFloor; };
#define MESH_COLOR Vec3(0.9f, 0.6f, 0.2f)

// ============================== host side: load + fit + build ==============================
std::vector<Triangle> tris;
std::vector<BVHNode>  nodes;
int bvhDepth = 0;      // only for the sanity check

bool loadOBJ(const char* path) {
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "Could not open %s\n", path); return false; }

    std::vector<Vec3> verts;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            verts.push_back(Vec3(x, y, z));
        } else if (tag == "f") {
            std::vector<int> idx;
            std::string tok;
            while (ss >> tok) {
                int i = std::stoi(tok);                      // reads the number before any '/'
                if (i < 0) i = (int)verts.size() + i + 1;    // negative = counted from the end
                idx.push_back(i - 1);                        // OBJ indices start at 1
            }
            for (size_t k = 1; k + 1 < idx.size(); k++)
                tris.push_back({ verts[idx[0]], verts[idx[k]], verts[idx[k + 1]] });
        }
    }
    printf("Loaded %s: %zu vertices, %zu triangles\n", path, verts.size(), tris.size());
    return true;
}

// Scale the mesh to `height` tall, centre it in x/z, and put its bottom at `base`.
void fitMesh(float height, Vec3 base) {
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (const Triangle& t : tris)
        for (Vec3 p : {t.v0, t.v1, t.v2}) {
            lo = Vec3(std::fmin(lo.x, p.x), std::fmin(lo.y, p.y), std::fmin(lo.z, p.z));
            hi = Vec3(std::fmax(hi.x, p.x), std::fmax(hi.y, p.y), std::fmax(hi.z, p.z));
        }
    float s = height / (hi.y - lo.y);
    Vec3 c((lo.x + hi.x) / 2, lo.y, (lo.z + hi.z) / 2);   // bottom-centre of the box
    for (Triangle& t : tris) {
        t.v0 = (t.v0 - c) * s + base;
        t.v1 = (t.v1 - c) * s + base;
        t.v2 = (t.v2 - c) * s + base;
    }
}

static float axisOf(Vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }
static float centroidOn(const Triangle& t, int a) {
    return (axisOf(t.v0, a) + axisOf(t.v1, a) + axisOf(t.v2, a)) / 3.0f;
}

// Builds the subtree for tris[first .. first+count) and returns its node index.
int build(int first, int count, int depth) {
    if (depth > bvhDepth) bvhDepth = depth;
    int me = (int)nodes.size();
    nodes.push_back(BVHNode());

    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (int i = first; i < first + count; i++)
        for (Vec3 p : {tris[i].v0, tris[i].v1, tris[i].v2}) {
            lo = Vec3(std::fmin(lo.x, p.x), std::fmin(lo.y, p.y), std::fmin(lo.z, p.z));
            hi = Vec3(std::fmax(hi.x, p.x), std::fmax(hi.y, p.y), std::fmax(hi.z, p.z));
        }
    nodes[me].bmin = lo;
    nodes[me].bmax = hi;

    if (count <= 4) {
        nodes[me].rightOrFirst = first;
        nodes[me].count = count;
        return me;
    }

    Vec3 ext = hi - lo;
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    float split = 0.5f * (axisOf(lo, axis) + axisOf(hi, axis));

    Triangle* base = tris.data();
    int mid = (int)(std::partition(base + first, base + first + count,
                    [&](const Triangle& t) { return centroidOn(t, axis) < split; }) - base);

    if (mid == first || mid == first + count) {
        mid = first + count / 2;
        std::nth_element(base + first, base + mid, base + first + count,
                         [&](const Triangle& a, const Triangle& b) {
                             return centroidOn(a, axis) < centroidOn(b, axis); });
    }

    build(first, mid - first, depth + 1);                        // left child lands at me + 1
    int right = build(mid, first + count - mid, depth + 1);
    nodes[me].rightOrFirst = right;
    nodes[me].count = 0;
    return me;
}

// ============================== device side: intersection + traversal ==============================
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

// Möller–Trumbore: solve origin + t*dir = v0 + u*e1 + v*e2 for t, u, v.
__device__ float hitTriangle(const Triangle& tr, Vec3 origin, Vec3 dir) {
    Vec3  e1 = tr.v1 - tr.v0, e2 = tr.v2 - tr.v0;
    Vec3  h = cross(dir, e2);
    float a = dot(e1, h);
    if (fabsf(a) < 1e-7f) return -1.0f;
    float f = 1.0f / a;
    Vec3  s = origin - tr.v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;
    Vec3  q = cross(s, e1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    float t = f * dot(e2, q);
    return (t > 0.001f) ? t : -1.0f;
}

// Slab test: does the ray enter the box before tMax?
__device__ bool hitBox(const BVHNode& b, Vec3 o, Vec3 inv, float tMax) {
    float t1 = (b.bmin.x - o.x) * inv.x, t2 = (b.bmax.x - o.x) * inv.x;
    float tmin = fminf(t1, t2), tmax = fmaxf(t1, t2);
    t1 = (b.bmin.y - o.y) * inv.y; t2 = (b.bmax.y - o.y) * inv.y;
    tmin = fmaxf(tmin, fminf(t1, t2)); tmax = fminf(tmax, fmaxf(t1, t2));
    t1 = (b.bmin.z - o.z) * inv.z; t2 = (b.bmax.z - o.z) * inv.z;
    tmin = fmaxf(tmin, fminf(t1, t2)); tmax = fminf(tmax, fmaxf(t1, t2));
    return tmax >= fmaxf(tmin, 0.0f) && tmin < tMax;
}

// Iterative traversal with a small per-thread stack (the GPU has no deep recursion).
// tBest comes in as the closest distance so far (the spheres) and goes out updated.
__device__ int bvhClosest(const Mesh& m, Vec3 origin, Vec3 dir, float& tBest) {
    Vec3 inv(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    int best = -1;
    int stack[64], sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int ni = stack[--sp];
        const BVHNode& nd = m.nodes[ni];
        if (!hitBox(nd, origin, inv, tBest)) continue;
        if (nd.count > 0) {
            for (int i = nd.rightOrFirst; i < nd.rightOrFirst + nd.count; i++) {
                float t = hitTriangle(m.tris[i], origin, dir);
                if (t > 0 && t < tBest) { tBest = t; best = i; }
            }
        } else {
            stack[sp++] = nd.rightOrFirst;   // right child
            stack[sp++] = ni + 1;            // left child, popped first
        }
    }
    return best;
}

__device__ bool closestHit(const Mesh& m, Vec3 origin, Vec3 dir, Hit& h) {
    h.t = 1e30f;
    int sph = -1, tri = -1;
    for (int i = 0; i < NUM_SPHERES; i++) {
        float t = hitSphere(scene[i], origin, dir);
        if (t > 0 && t < h.t) { h.t = t; sph = i; tri = -1; }
    }
#ifdef BRUTE
    for (int i = 0; i < m.numTris; i++) {
        float t = hitTriangle(m.tris[i], origin, dir);
        if (t > 0 && t < h.t) { h.t = t; tri = i; sph = -1; }
    }
#else
    float tb = h.t;
    int bi = bvhClosest(m, origin, dir, tb);
    if (bi >= 0) { h.t = tb; tri = bi; sph = -1; }
#endif
    if (sph < 0 && tri < 0) return false;

    Vec3 p = origin + dir * h.t;
    if (sph >= 0) {
        h.n = normalize(p - scene[sph].center);
        h.color = scene[sph].color;  h.k = scene[sph].reflectivity;  h.isFloor = scene[sph].isFloor;
    } else {
        const Triangle& tr = m.tris[tri];
        Vec3 n = normalize(cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
        if (dot(n, dir) > 0) n = n * -1.0f;      // face the ray
        h.n = n;  h.color = MESH_COLOR;  h.k = BALL_K;  h.isFloor = false;
    }
    return true;
}

// Loop version of the reflections: color + weight, no recursion.
__device__ Vec3 rayColorIterative(const Mesh& m, Vec3 origin, Vec3 dir, int bounces) {
    Vec3  lightPos(4, 6, 2);
    Vec3  ans(0.0f, 0.0f, 0.0f);
    float weight = 1.0f;

    bounces++;
    while (bounces--) {
        Hit h;
        if (!closestHit(m, origin, dir, h)) {
            float t = 0.5f * (dir.y + 1.0f);
            Vec3 sky = Vec3(1, 1, 1) * (1 - t) + Vec3(0.4f, 0.6f, 1.0f) * t;
            return ans + sky * weight;
        }

        Vec3 p = origin + dir * h.t;
        Vec3 n = h.n;

        Vec3 base = h.color;
        if (h.isFloor) {
            int cx = (int)floorf(p.x), cz = (int)floorf(p.z);
            bool white = ((cx + cz) & 1) == 0;
            base = white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f);
        }

        Vec3  toLight = normalize(lightPos - p);
        float diffuse = fmaxf(0.0f, dot(n, toLight));
        Hit   sh;
        bool  blocked = closestHit(m, p + n * 0.001f, toLight, sh);
        float shade = blocked ? 0.3f : 1.0f;
        Vec3  local_color = base * (0.15f + diffuse * shade);

        float k = h.k;
        if (bounces == 0 || k == 0) return ans + local_color * weight;

        float cosTheta = fmaxf(0.0f, dot(n, dir * -1.0f));
        float x  = 1.0f - cosTheta;
        float x2 = x * x;
        float F  = k + (1.0f - k) * (x2 * x2 * x);   // x^5 without pow

        ans    = ans + local_color * ((1.0f - F) * weight);
        weight = weight * F;
        origin = p + n * 0.001f;
        dir    = dir - n * (2.0f * dot(dir, n));
    }
    return ans;
}

__global__ void render(unsigned char* image, int W, int H, Mesh m) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;   // grid is rounded up past the edge

    Vec3  cameraPos(0, 1, 3);
    float aspect = (float)W / H;

    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3  dir = normalize(Vec3(u, v, -1.5f));

    Vec3 color = rayColorIterative(m, cameraPos, dir, NUM_BOUNCES);

    int idx = (py * W + px) * 3;
    image[idx + 0] = (unsigned char)(fminf(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
}

int main() {
    if (!loadOBJ("stanford-bunny.obj")) return 1;
    fitMesh(1.0f, Vec3(0, -0.5f, -1.0f));   // same placement as the CPU file, so the images can be compared

    auto b0 = std::chrono::high_resolution_clock::now();
    build(0, (int)tris.size(), 1);          // reorders `tris`, so build BEFORE copying to the GPU
    auto b1 = std::chrono::high_resolution_clock::now();
    printf("BVH: %zu nodes, depth %d, build %.1f ms\n", nodes.size(), bvhDepth,
           std::chrono::duration<double, std::milli>(b1 - b0).count());

    const int W = WIDTH, H = HEIGHT;
    std::vector<unsigned char> image(W * H * 3);

    Sphere hostScene[NUM_SPHERES] = {
        { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), FLOOR_K, true  }, // floor
#ifndef NO_BALLS
        { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), BALL_K,  false }, // red
        { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), BALL_K,  false }, // green
        { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), BALL_K,  false }, // blue
#endif
    };
    CUDA_CHECK(cudaMemcpyToSymbol(scene, hostScene, sizeof(hostScene)));

    // Mesh + BVH go to global memory.
    Triangle* d_tris;  BVHNode* d_nodes;  unsigned char* d_image;
    CUDA_CHECK(cudaMalloc(&d_tris,  tris.size()  * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&d_nodes, nodes.size() * sizeof(BVHNode)));
    CUDA_CHECK(cudaMalloc(&d_image, image.size()));
    CUDA_CHECK(cudaMemcpy(d_tris,  tris.data(),  tris.size()  * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(BVHNode),  cudaMemcpyHostToDevice));
    Mesh mesh = { d_tris, d_nodes, (int)tris.size() };

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // warm-up launch (not timed): the first launch pays one-time setup costs
    render<<<grid, block>>>(d_image, W, H, mesh);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, mid, stop;
    cudaEventCreate(&start); cudaEventCreate(&mid); cudaEventCreate(&stop);

    cudaEventRecord(start);
    render<<<grid, block>>>(d_image, W, H, mesh);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(mid);
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, image.size(), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0, totalMs = 0;
    cudaEventElapsedTime(&kernelMs, start, mid);
    cudaEventElapsedTime(&totalMs, start, stop);
    printf("kernel: %.3f ms   total (incl. copy back): %.3f ms\n", kernelMs, totalMs);

    cudaFree(d_image); cudaFree(d_tris); cudaFree(d_nodes);

    const char* outPath = "../images/output/gpu_bvh.ppm";
    FILE* f = fopen(outPath, "wb");
    if (!f) {
        fprintf(stderr, "Could not open %s (run: mkdir -p ../images/output)\n", outPath);
        return 1;
    }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image.data(), 1, image.size(), f);
    fclose(f);
    printf("Wrote %s (%dx%d)\n", outPath, W, H);
    return 0;
}
