// Experiment build of the GPU mesh + BVH ray tracer (raytracer_gpu_bvh.cu with switches).
//
// COMPILE-TIME switches (nvcc -D...):
//   -DBLOCK_X=16 -DBLOCK_Y=16   block shape (default 16x16)
//   -DSTACK_SIZE=64             traversal stack entries per thread (default 64; the tree depth must fit)
//   -DLAYOUT=0|1                0 = original structs; 1 = 32-byte nodes and 48-byte triangles read as float4
//   -DBRUTE                     test every triangle instead of using the BVH
//   -DNO_BALLS                  remove the three balls (as in the CPU runs)
//   -DSCENE=n                   as in config.h
//
// RUN-TIME arguments (all optional; -1 keeps the default):
//   gpu_exp.exe [W] [H] [bounces] [copies] [runs] [heat]
//   copies = number of bunnies laid out on a grid (scene-size sweep); runs = timed launches (median is reported)
//   heat   = 1 also writes ../images/output/heat_nodes.ppm and heat_tris.ppm (divergence / work heat maps)
//
// Prints one machine-readable line starting with RESULT, which run_experiments.ps1 collects.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include "config.h"

#ifndef BLOCK_X
#define BLOCK_X 16
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif
#ifndef STACK_SIZE
#define STACK_SIZE 64
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif

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
    bool  isFloor;
};

#ifdef NO_BALLS
#define NUM_SPHERES 1
#else
#define NUM_SPHERES 4
#endif
__constant__ Sphere scene[NUM_SPHERES];

// ---- host-side structures (what the OBJ loader and the BVH build produce) ----
struct Triangle { Vec3 v0, v1, v2; };
struct BVHNode {
    Vec3 bmin, bmax;
    int  rightOrFirst; // inner node: index of the RIGHT child (the left child is the next node); leaf: first triangle
    int  count;        // 0 = inner node; >0 = leaf with this many triangles
};

// ---- device-side storage layouts ----
// LAYOUT 1: a node is two float4 (32 bytes, two aligned 16-byte loads); the integer fields ride in the w slots.
//           a triangle is v0, e1 = v1-v0, e2 = v2-v0, each padded to a float4 (48 bytes).
struct alignas(16) NodeV { float4 lo; float4 hi; };
struct alignas(16) TriV  { float4 v0; float4 e1; float4 e2; };

// What the traversal code works with, whichever layout is in memory.
struct NodeIn { Vec3 bmin, bmax; int first; int count; };
struct TriIn  { Vec3 v0, e1, e2; };

#if LAYOUT == 1
typedef NodeV NodeT;
typedef TriV  TriT;
__device__ NodeIn loadNode(const NodeT* p, int i) {
    float4 a = p[i].lo, b = p[i].hi;
    NodeIn n;
    n.bmin = Vec3(a.x, a.y, a.z);
    n.bmax = Vec3(b.x, b.y, b.z);
    n.first = __float_as_int(a.w);
    n.count = __float_as_int(b.w);
    return n;
}
__device__ TriIn loadTri(const TriT* p, int i) {
    float4 a = p[i].v0, b = p[i].e1, c = p[i].e2;
    TriIn t;
    t.v0 = Vec3(a.x, a.y, a.z);
    t.e1 = Vec3(b.x, b.y, b.z);
    t.e2 = Vec3(c.x, c.y, c.z);
    return t;
}
#else
typedef BVHNode  NodeT;
typedef Triangle TriT;
__device__ NodeIn loadNode(const NodeT* p, int i) {
    const BVHNode& s = p[i];
    NodeIn n;
    n.bmin = s.bmin; n.bmax = s.bmax; n.first = s.rightOrFirst; n.count = s.count;
    return n;
}
__device__ TriIn loadTri(const TriT* p, int i) {
    const Triangle& s = p[i];
    TriIn t;
    t.v0 = s.v0; t.e1 = s.v1 - s.v0; t.e2 = s.v2 - s.v0;
    return t;
}
#endif

struct Mesh {
    const TriT*  tris;
    const NodeT* nodes;
    int          numTris;
};

struct Hit { float t; Vec3 n; Vec3 color; float k; bool isFloor; };
#define MESH_COLOR Vec3(0.9f, 0.6f, 0.2f)

// ============================== host side: load + fit + build ==============================
std::vector<Triangle> tris;
std::vector<BVHNode>  nodes;
int bvhDepth = 0;

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
                int i = std::stoi(tok);
                if (i < 0) i = (int)verts.size() + i + 1;
                idx.push_back(i - 1);
            }
            for (size_t k = 1; k + 1 < idx.size(); k++)
                tris.push_back({ verts[idx[0]], verts[idx[k]], verts[idx[k + 1]] });
        }
    }
    printf("Loaded %s: %zu vertices, %zu triangles\n", path, verts.size(), tris.size());
    return true;
}

void fitMesh(float height, Vec3 base) {
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (const Triangle& t : tris)
        for (Vec3 p : {t.v0, t.v1, t.v2}) {
            lo = Vec3(std::fmin(lo.x, p.x), std::fmin(lo.y, p.y), std::fmin(lo.z, p.z));
            hi = Vec3(std::fmax(hi.x, p.x), std::fmax(hi.y, p.y), std::fmax(hi.z, p.z));
        }
    float s = height / (hi.y - lo.y);
    Vec3 c((lo.x + hi.x) / 2, lo.y, (lo.z + hi.z) / 2);
    for (Triangle& t : tris) {
        t.v0 = (t.v0 - c) * s + base;
        t.v1 = (t.v1 - c) * s + base;
        t.v2 = (t.v2 - c) * s + base;
    }
}

// Scene-size sweep: lay `copies` bunnies out on a grid (in x and further back in z).
void replicateMesh(int copies) {
    if (copies <= 1) return;
    std::vector<Triangle> one = tris;
    int cols = (int)std::ceil(std::sqrt((double)copies));
    tris.clear();
    tris.reserve(one.size() * copies);
    for (int k = 0; k < copies; k++) {
        float dx = ((k % cols) - (cols - 1) / 2.0f) * 1.2f;
        float dz = -(float)(k / cols) * 1.2f;
        Vec3 d(dx, 0.0f, dz);
        for (const Triangle& t : one) tris.push_back({ t.v0 + d, t.v1 + d, t.v2 + d });
    }
}

static float axisOf(Vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }
static float centroidOn(const Triangle& t, int a) {
    return (axisOf(t.v0, a) + axisOf(t.v1, a) + axisOf(t.v2, a)) / 3.0f;
}

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

    build(first, mid - first, depth + 1);
    int right = build(mid, first + count - mid, depth + 1);
    nodes[me].rightOrFirst = right;
    nodes[me].count = 0;
    return me;
}

static float bitsToFloat(int v) { float f; std::memcpy(&f, &v, sizeof f); return f; }

// ============================== device side ==============================
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

// Möller–Trumbore, with the edges e1, e2 already known.
__device__ float hitTriangle(const TriIn& tr, Vec3 origin, Vec3 dir) {
    Vec3  h = cross(dir, tr.e2);
    float a = dot(tr.e1, h);
    if (fabsf(a) < 1e-7f) return -1.0f;
    float f = 1.0f / a;
    Vec3  s = origin - tr.v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;
    Vec3  q = cross(s, tr.e1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    float t = f * dot(tr.e2, q);
    return (t > 0.001f) ? t : -1.0f;
}

__device__ bool hitBox(const NodeIn& b, Vec3 o, Vec3 inv, float tMax) {
    float t1 = (b.bmin.x - o.x) * inv.x, t2 = (b.bmax.x - o.x) * inv.x;
    float tmin = fminf(t1, t2), tmax = fmaxf(t1, t2);
    t1 = (b.bmin.y - o.y) * inv.y; t2 = (b.bmax.y - o.y) * inv.y;
    tmin = fmaxf(tmin, fminf(t1, t2)); tmax = fminf(tmax, fmaxf(t1, t2));
    t1 = (b.bmin.z - o.z) * inv.z; t2 = (b.bmax.z - o.z) * inv.z;
    tmin = fmaxf(tmin, fminf(t1, t2)); tmax = fminf(tmax, fmaxf(t1, t2));
    return tmax >= fmaxf(tmin, 0.0f) && tmin < tMax;
}

__device__ int bvhClosest(const Mesh& m, Vec3 origin, Vec3 dir, float& tBest) {
    Vec3 inv(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    int best = -1;
    int stack[STACK_SIZE], sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int ni = stack[--sp];
        NodeIn nd = loadNode(m.nodes, ni);
        if (!hitBox(nd, origin, inv, tBest)) continue;
        if (nd.count > 0) {
            for (int i = nd.first; i < nd.first + nd.count; i++) {
                float t = hitTriangle(loadTri(m.tris, i), origin, dir);
                if (t > 0 && t < tBest) { tBest = t; best = i; }
            }
        } else {
            stack[sp++] = nd.first;   // right child
            stack[sp++] = ni + 1;     // left child, popped first
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
        float t = hitTriangle(loadTri(m.tris, i), origin, dir);
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
        TriIn tr = loadTri(m.tris, tri);
        Vec3 n = normalize(cross(tr.e1, tr.e2));
        if (dot(n, dir) > 0) n = n * -1.0f;
        h.n = n;  h.color = MESH_COLOR;  h.k = BALL_K;  h.isFloor = false;
    }
    return true;
}

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
        float F  = k + (1.0f - k) * (x2 * x2 * x);

        ans    = ans + local_color * ((1.0f - F) * weight);
        weight = weight * F;
        origin = p + n * 0.001f;
        dir    = dir - n * (2.0f * dot(dir, n));
    }
    return ans;
}

__global__ void render(unsigned char* image, int W, int H, int bounces, Mesh m) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    Vec3  cameraPos(0, 1, 3);
    float aspect = (float)W / H;

    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3  dir = normalize(Vec3(u, v, -1.5f));

    Vec3 color = rayColorIterative(m, cameraPos, dir, bounces);

    int idx = (py * W + px) * 3;
    image[idx + 0] = (unsigned char)(fminf(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fminf(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fminf(color.z, 1.0f) * 255);
}

// Heat map kernel: for the PRIMARY ray of each pixel, count the BVH nodes visited and triangles tested.
// (Same traversal as bvhClosest, no early cut from spheres.)
__global__ void heat(int* out, int W, int H, Mesh m) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    Vec3  origin(0, 1, 3);
    float aspect = (float)W / H;
    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3  dir = normalize(Vec3(u, v, -1.5f));

    Vec3 inv(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    float tBest = 1e30f;
    int nodesVisited = 0, trisTested = 0;
    int stack[STACK_SIZE], sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        int ni = stack[--sp];
        NodeIn nd = loadNode(m.nodes, ni);
        nodesVisited++;
        if (!hitBox(nd, origin, inv, tBest)) continue;
        if (nd.count > 0) {
            for (int i = nd.first; i < nd.first + nd.count; i++) {
                trisTested++;
                float t = hitTriangle(loadTri(m.tris, i), origin, dir);
                if (t > 0 && t < tBest) tBest = t;
            }
        } else {
            stack[sp++] = nd.first;
            stack[sp++] = ni + 1;
        }
    }
    int idx = (py * W + px) * 2;
    out[idx + 0] = nodesVisited;
    out[idx + 1] = trisTested;
}

// ============================== host helpers + main ==============================
static bool writePPM(const char* path, int W, int H, const unsigned char* data) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Could not open %s (run: mkdir -p ../images/output)\n", path); return false; }
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(data, 1, (size_t)W * H * 3, f);
    fclose(f);
    printf("Wrote %s (%dx%d)\n", path, W, H);
    return true;
}

static void colormap(float t, unsigned char* rgb) {          // 0..1 -> blue, cyan, green, yellow, red
    auto ramp = [](float v) { return std::fmin(1.0f, std::fmax(0.0f, v)); };
    rgb[0] = (unsigned char)(255 * ramp(1.5f - std::fabs(4 * t - 3)));
    rgb[1] = (unsigned char)(255 * ramp(1.5f - std::fabs(4 * t - 2)));
    rgb[2] = (unsigned char)(255 * ramp(1.5f - std::fabs(4 * t - 1)));
}

static void writeHeat(const char* path, const std::vector<int>& h, int which, int W, int H, double& mean, int& mx) {
    mx = 0; double sum = 0;
    for (int i = 0; i < W * H; i++) { mx = std::max(mx, h[2 * i + which]); sum += h[2 * i + which]; }
    mean = sum / (W * H);
    std::vector<unsigned char> img((size_t)W * H * 3);
    for (int i = 0; i < W * H; i++)
        colormap(mx > 0 ? (float)h[2 * i + which] / mx : 0.0f, &img[3 * (size_t)i]);
    writePPM(path, W, H, img.data());
}

static int argOr(int argc, char** argv, int i, int def) {
    if (argc > i) { int v = atoi(argv[i]); if (v >= 0) return v; }
    return def;
}

int main(int argc, char** argv) {
    const int W = argOr(argc, argv, 1, WIDTH);
    const int H = argOr(argc, argv, 2, HEIGHT);
    const int bounces = argOr(argc, argv, 3, NUM_BOUNCES);
    const int copies = std::max(1, argOr(argc, argv, 4, 1));
    const int runs = std::max(1, argOr(argc, argv, 5, 5));
    const int doHeat = argOr(argc, argv, 6, 0);

    if (!loadOBJ("stanford-bunny.obj")) return 1;
    fitMesh(1.0f, Vec3(0, -0.5f, -1.0f));
    replicateMesh(copies);

    auto b0 = std::chrono::high_resolution_clock::now();
    build(0, (int)tris.size(), 1);          // reorders `tris`, so build BEFORE copying to the GPU
    auto b1 = std::chrono::high_resolution_clock::now();
    printf("BVH: %zu nodes, depth %d, build %.1f ms\n", nodes.size(), bvhDepth,
           std::chrono::duration<double, std::milli>(b1 - b0).count());
    if (bvhDepth > STACK_SIZE) {
        fprintf(stderr, "Tree depth %d does not fit STACK_SIZE %d: rebuild with a larger -DSTACK_SIZE\n", bvhDepth, STACK_SIZE);
        return 1;
    }

    std::vector<unsigned char> image((size_t)W * H * 3);

    Sphere hostScene[NUM_SPHERES] = {
        { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), FLOOR_K, true  },
#ifndef NO_BALLS
        { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), BALL_K,  false },
        { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), BALL_K,  false },
        { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), BALL_K,  false },
#endif
    };
    CUDA_CHECK(cudaMemcpyToSymbol(scene, hostScene, sizeof(hostScene)));

    // Convert to the device layout and upload.
#if LAYOUT == 1
    std::vector<NodeV> hn(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        const BVHNode& n = nodes[i];
        hn[i].lo = make_float4(n.bmin.x, n.bmin.y, n.bmin.z, bitsToFloat(n.rightOrFirst));
        hn[i].hi = make_float4(n.bmax.x, n.bmax.y, n.bmax.z, bitsToFloat(n.count));
    }
    std::vector<TriV> ht(tris.size());
    for (size_t i = 0; i < tris.size(); i++) {
        const Triangle& t = tris[i];
        Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
        ht[i].v0 = make_float4(t.v0.x, t.v0.y, t.v0.z, 0.0f);
        ht[i].e1 = make_float4(e1.x, e1.y, e1.z, 0.0f);
        ht[i].e2 = make_float4(e2.x, e2.y, e2.z, 0.0f);
    }
#else
    const std::vector<BVHNode>& hn = nodes;
    const std::vector<Triangle>& ht = tris;
#endif
    TriT* d_tris;  NodeT* d_nodes;  unsigned char* d_image;
    CUDA_CHECK(cudaMalloc(&d_tris,  ht.size() * sizeof(TriT)));
    CUDA_CHECK(cudaMalloc(&d_nodes, hn.size() * sizeof(NodeT)));
    CUDA_CHECK(cudaMalloc(&d_image, image.size()));
    CUDA_CHECK(cudaMemcpy(d_tris,  ht.data(), ht.size() * sizeof(TriT),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nodes, hn.data(), hn.size() * sizeof(NodeT), cudaMemcpyHostToDevice));
    Mesh mesh = { d_tris, d_nodes, (int)tris.size() };

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    // warm-up launch (not timed)
    render<<<grid, block>>>(d_image, W, H, bounces, mesh);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> times;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(e0);
        render<<<grid, block>>>(d_image, W, H, bounces, mesh);
        cudaEventRecord(e1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    float median = times[times.size() / 2];

    cudaEventRecord(e0);
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, image.size(), cudaMemcpyDeviceToHost));
    cudaEventRecord(e1);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float copyMs = 0;
    cudaEventElapsedTime(&copyMs, e0, e1);

    unsigned long long hash = 14695981039346656037ULL;      // FNV-1a over the image, to spot any change in output
    for (unsigned char c : image) { hash ^= c; hash *= 1099511628211ULL; }

    printf("kernel: median %.3f ms (min %.3f, max %.3f over %d runs)   copy back: %.3f ms\n",
           median, times.front(), times.back(), runs, copyMs);
    printf("RESULT,block=%dx%d,layout=%d,stack=%d,W=%d,H=%d,bounces=%d,copies=%d,tris=%zu,nodes=%zu,depth=%d,median_ms=%.3f,min_ms=%.3f,max_ms=%.3f,copy_ms=%.3f,checksum=%016llx\n",
           BLOCK_X, BLOCK_Y, LAYOUT, STACK_SIZE, W, H, bounces, copies, tris.size(), nodes.size(), bvhDepth,
           median, times.front(), times.back(), copyMs, hash);

    writePPM("../images/output/gpu_exp.ppm", W, H, image.data());

    if (doHeat) {
        int* d_heat;
        CUDA_CHECK(cudaMalloc(&d_heat, (size_t)W * H * 2 * sizeof(int)));
        heat<<<grid, block>>>(d_heat, W, H, mesh);
        CUDA_CHECK(cudaGetLastError());
        std::vector<int> h((size_t)W * H * 2);
        CUDA_CHECK(cudaMemcpy(h.data(), d_heat, h.size() * sizeof(int), cudaMemcpyDeviceToHost));
        double nm, tm; int nx, tx;
        writeHeat("../images/output/heat_nodes.ppm", h, 0, W, H, nm, nx);
        writeHeat("../images/output/heat_tris.ppm",  h, 1, W, H, tm, tx);
        printf("HEAT,nodes_mean=%.2f,nodes_max=%d,tris_mean=%.2f,tris_max=%d\n", nm, nx, tm, tx);
        cudaFree(d_heat);
    }

    cudaFree(d_image); cudaFree(d_tris); cudaFree(d_nodes);
    return 0;
}
