// ============================================================================
//  Basic CPU ray tracer  --  starter code
// ============================================================================
//  Renders a few spheres on a checkerboard floor, with simple diffuse shading
//  and hard shadows. Single-threaded, plain C++. No external libraries.
//
//  This is your STARTING POINT -- it is short on purpose.
//  Later you will (a) add reflections, (b) port it to CUDA, and (c) make it fast.
//
//  Build:  g++ -O2 -o raytracer_cpu starter_raytracer_cpu.cpp
//  Run:    ./raytracer_cpu                 (writes image.ppm)
//  View:   most image viewers open .ppm; or convert with:  convert image.ppm image.png
// ============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream> 
#include <string>
#include <algorithm>
#include "config.h"


// ---- A tiny 3D vector, used for points, directions, and colors (r,g,b) -----
struct Vec3 {
    float x, y, z;
    Vec3(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
};
Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
float dot(Vec3 a, Vec3 b)  { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3  normalize(Vec3 a)    { float len = std::sqrt(dot(a, a)); return a * (1.0f / len); }
Vec3 cross(Vec3 a, Vec3 b) { 
    float i1 = a.x, j1 = a.y, k1 = a.z;
    float i2 = b.x, j2 = b.y, k2 = b.z;
    float i_new = j1*k2 - j2*k1;
    float j_new = -i1*k2 + i2*k1;
    float k_new = i1*j2 - i2*j1;
    return Vec3(i_new, j_new, k_new);
}

// ---- A sphere: a center, a radius, and a color ----------------------------
struct Sphere {
    Vec3  center;
    float radius;
    Vec3  color;
    float reflectivity;
    bool  isFloor;   // if true, we draw a checkerboard instead of a solid color
};

struct Triangle { Vec3 v0, v1, v2; };
std::vector<Triangle> tris;
struct Hit { float t; Vec3 n; Vec3 color; float k; bool isFloor; };


struct BVHNode {
    Vec3 bmin, bmax;   // box around everything below this node
    int  rightOrFirst; // inner node: index of the RIGHT child (the left child is the next node)
                       // leaf: index of its first triangle in `tris`
    int  count;        // 0 = inner node; >0 = leaf with this many triangles
};
std::vector<BVHNode> nodes;




// The whole scene is just a list of spheres. (The floor is a giant sphere.)
std::vector<Sphere> scene = {
    { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), FLOOR_K, true }, // floor
#ifndef NO_BALLS
    { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), BALL_K, false }, // red
    { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), BALL_K, false }, // green
    { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), BALL_K, false }, // blue
#endif
};

// A ray is a point p(t) = origin + t * dir. This finds where (if anywhere) the
// ray hits the sphere, returning the nearest distance t in front of the camera,
// or -1 if it misses. (Solves a quadratic -- see any ray-sphere reference.)
float hitSphere(const Sphere& s, Vec3 origin, Vec3 dir) {
    Vec3  oc = origin - s.center;
    float a = dot(dir, dir);
    float b = 2.0f * dot(oc, dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0) return -1.0f;                 // no real solution -> miss
    float t = (-b - std::sqrt(disc)) / (2 * a); // nearest intersection
    return (t > 0.001f) ? t : -1.0f;
}

// Möller–Trumbore: solve origin + t*dir = v0 + u*e1 + v*e2 for t, u, v.
float hitTriangle(const Triangle& tr, Vec3 origin, Vec3 dir) {
    Vec3  e1 = tr.v1 - tr.v0, e2 = tr.v2 - tr.v0;
    Vec3  h = cross(dir, e2);
    float a = dot(e1, h);
    if (std::fabs(a) < 1e-7f) return -1.0f;      // ray parallel to the triangle
    float f = 1.0f / a;
    Vec3  s = origin - tr.v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return -1.0f;
    Vec3  q = cross(s, e1);
    float v = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return -1.0f;  // outside the triangle
    float t = f * dot(e2, q);
    return (t > 0.001f) ? t : -1.0f;
}
// Slab test: does the ray enter this box before distance tMax?
static inline bool hitBox(const BVHNode& b, Vec3 o, Vec3 inv, float tMax) {
    float t1 = (b.bmin.x - o.x) * inv.x, t2 = (b.bmax.x - o.x) * inv.x;
    float tmin = std::fmin(t1, t2), tmax = std::fmax(t1, t2);
    t1 = (b.bmin.y - o.y) * inv.y; t2 = (b.bmax.y - o.y) * inv.y;
    tmin = std::fmax(tmin, std::fmin(t1, t2)); tmax = std::fmin(tmax, std::fmax(t1, t2));
    t1 = (b.bmin.z - o.z) * inv.z; t2 = (b.bmax.z - o.z) * inv.z;
    tmin = std::fmax(tmin, std::fmin(t1, t2)); tmax = std::fmin(tmax, std::fmax(t1, t2));
    return tmax >= std::fmax(tmin, 0.0f) && tmin < tMax;
}

// Closest triangle along the ray, using the tree. Returns its index (or -1); tBest is the
// distance to beat on the way in, and the hit distance on the way out.
int bvhClosest(Vec3 origin, Vec3 dir, float& tBest) {
    Vec3 inv(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    int best = -1;
    int stack[64], sp = 0;
    stack[sp++] = 0;                                  // start at the root
    while (sp > 0) {
        int ni = stack[--sp];
        const BVHNode& nd = nodes[ni];
        if (!hitBox(nd, origin, inv, tBest)) continue;    // missed the box: skip everything inside
        if (nd.count > 0) {                               // leaf: test its few triangles
            for (int i = nd.rightOrFirst; i < nd.rightOrFirst + nd.count; i++) {
                float t = hitTriangle(tris[i], origin, dir);
                if (t > 0 && t < tBest) { tBest = t; best = i; }
            }
        } else {                                          // inner node: visit both children
            stack[sp++] = nd.rightOrFirst;                // right child
            stack[sp++] = ni + 1;                         // left child, popped first
        }
    }
    return best;
}


// Find the closest sphere the ray hits. Returns its index (or -1) and the distance.
const Vec3 MESH_COLOR(0.9f, 0.6f, 0.2f);

bool closestHit(Vec3 origin, Vec3 dir, Hit& h) {
    h.t = 1e30f;
    int sph = -1, tri = -1;
    for (int i = 0; i < (int)scene.size(); i++) {
        float t = hitSphere(scene[i], origin, dir);
        if (t > 0 && t < h.t) { h.t = t; sph = i; tri = -1; }
    }
    #ifdef BRUTE
        for (int i = 0; i < (int)tris.size(); i++) {
            float t = hitTriangle(tris[i], origin, dir);
            if (t > 0 && t < h.t) { h.t = t; tri = i; sph = -1; }
        }
    #else
        float tb = h.t;                       // a sphere hit already closer than this culls boxes
        int bi = bvhClosest(origin, dir, tb);
        if (bi >= 0) { h.t = tb; tri = bi; sph = -1; }
    #endif
    if (sph < 0 && tri < 0) return false;

    Vec3 p = origin + dir * h.t;
    if (sph >= 0) {
        h.n = normalize(p - scene[sph].center);
        h.color = scene[sph].color;  h.k = scene[sph].reflectivity;  h.isFloor = scene[sph].isFloor;
    } else {
        const Triangle& tr = tris[tri];
        Vec3 n = normalize(cross(tr.v1 - tr.v0, tr.v2 - tr.v0));
        if (dot(n, dir) > 0) n = n * -1.0f;      // face the ray
        h.n = n;  h.color = MESH_COLOR;  h.k = BALL_K;  h.isFloor = false;
    }
    return true;
}



// Compute the color seen along one ray.
// Compute the color seen along one ray (loop version: color + weight).
Vec3 rayColorIterative(Vec3 origin, Vec3 dir, int bounces) {
    Vec3  lightPos(4, 6, 2);
    Vec3  ans(0.0f, 0.0f, 0.0f);
    float weight = 1.0f;

    bounces++;
    while (bounces--) {
        Hit h;
        if (!closestHit(origin, dir, h)) {
            // Miss: sky gradient (white near horizon -> blue up top).
            float t = 0.5f * (dir.y + 1.0f);
            Vec3 sky = Vec3(1, 1, 1) * (1 - t) + Vec3(0.4f, 0.6f, 1.0f) * t;
            return ans + sky * weight;
        }

        Vec3 p = origin + dir * h.t;   // hit point
        Vec3 n = h.n;                  // surface normal (sphere or triangle)

        // Base color: checkerboard for the floor, solid color otherwise.
        Vec3 base = h.color;
        if (h.isFloor) {
            int cx = (int)std::floor(p.x), cz = (int)std::floor(p.z);
            bool white = ((cx + cz) & 1) == 0;
            base = white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f);
        }

        // Diffuse lighting + hard shadow.
        Vec3  toLight = normalize(lightPos - p);
        float diffuse = std::fmax(0.0f, dot(n, toLight));
        Hit   sh;
        bool  blocked = closestHit(p + n * 0.001f, toLight, sh);
        float shade = blocked ? 0.3f : 1.0f;
        Vec3  local_color = base * (0.15f + diffuse * shade);   // 0.15 = ambient

        float k = h.k;
        if (bounces == 0 || k == 0) return ans + local_color * weight;

        // Fresnel (Schlick): more reflective at grazing angles.
        float cosTheta = std::fmax(0.0f, dot(n, dir * -1.0f));
        float x  = 1.0f - cosTheta;
        float x2 = x * x;
        float F  = k + (1.0f - k) * (x2 * x2 * x);   // x^5 without pow

        // Continue with the reflected ray.
        ans    = ans + local_color * ((1.0f - F) * weight);
        weight = weight * F;
        origin = p + n * 0.001f;
        dir    = dir - n * (2.0f * dot(dir, n));
    }
    return ans;
}


// Load a Wavefront .obj file into `tris`. Only "v" (vertex) and "f" (face) lines matter.
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
                int i = std::stoi(tok);              // reads the number before any '/'
                if (i < 0) i = (int)verts.size() + i + 1;   // negative = counted from the end
                idx.push_back(i - 1);                // OBJ indices start at 1
            }
            // split a face with 3+ corners into triangles: (0,1,2), (0,2,3), ...
            for (size_t k = 1; k + 1 < idx.size(); k++)
                tris.push_back({ verts[idx[0]], verts[idx[k]], verts[idx[k + 1]] });
        }
    }
    printf("Loaded %s: %zu vertices, %zu triangles\n", path, verts.size(), tris.size());
    return true;
}



int bvhDepth = 0;      // only for the sanity check

static float axisOf(Vec3 v, int a) { return a == 0 ? v.x : (a == 1 ? v.y : v.z); }
static float centroidOn(const Triangle& t, int a) {
    return (axisOf(t.v0, a) + axisOf(t.v1, a) + axisOf(t.v2, a)) / 3.0f;
}

// Builds the subtree for tris[first .. first+count) and returns its node index.
int build(int first, int count, int depth) {
    if (depth > bvhDepth) bvhDepth = depth;
    int me = (int)nodes.size();
    nodes.push_back(BVHNode());              // reserve my slot (always use the index, not a reference)

    // Bounding box of all the triangles here.
    Vec3 lo(1e30f, 1e30f, 1e30f), hi(-1e30f, -1e30f, -1e30f);
    for (int i = first; i < first + count; i++)
        for (Vec3 p : {tris[i].v0, tris[i].v1, tris[i].v2}) {
            lo = Vec3(std::fmin(lo.x, p.x), std::fmin(lo.y, p.y), std::fmin(lo.z, p.z));
            hi = Vec3(std::fmax(hi.x, p.x), std::fmax(hi.y, p.y), std::fmax(hi.z, p.z));
        }
    nodes[me].bmin = lo;
    nodes[me].bmax = hi;

    // Few triangles left: make a leaf.
    if (count <= 4) {
        nodes[me].rightOrFirst = first;
        nodes[me].count = count;
        return me;
    }

    // Split on the longest axis, at the middle of the box.
    Vec3 ext = hi - lo;
    int axis = (ext.x > ext.y && ext.x > ext.z) ? 0 : (ext.y > ext.z ? 1 : 2);
    float split = 0.5f * (axisOf(lo, axis) + axisOf(hi, axis));

    // Triangles whose centre is below the split go first; mid is where the second group starts.
    Triangle* base = tris.data();
    int mid = (int)(std::partition(base + first, base + first + count,
                    [&](const Triangle& t) { return centroidOn(t, axis) < split; }) - base);

    // All triangles landed on one side: split at the median instead, or the recursion never ends.
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

int main() {
    if (!loadOBJ("stanford-bunny.obj")) return 1;
    fitMesh(1.0f, Vec3(0, -0.5f, -1.0f));

    auto b0 = std::chrono::high_resolution_clock::now();
    build(0, (int)tris.size(), 1);
    auto b1 = std::chrono::high_resolution_clock::now();
    printf("BVH: %zu nodes, depth %d, build %.1f ms\n", nodes.size(), bvhDepth,
           std::chrono::duration<double, std::milli>(b1 - b0).count());


    const int W = WIDTH, H = HEIGHT; //Setup image dimensions
    std::vector<unsigned char> image(W * H * 3);//Total Pixels - Stored as (R, G, B per pixel)

    // TRY IT: move the camera and see how the view changes.
    // x = right, y = up, z = toward the viewer (spheres sit near z = -1).
    Vec3  cameraPos(0, 1, 3);
    float aspect = (float)W / H;


    auto t0 = std::chrono::high_resolution_clock::now();
    // For every pixel, build a ray from the camera and compute its color.
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            // Map pixel (px,py) to a direction through the image plane.
            float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
            float v = (1.0f - 2.0f * (py + 0.5f) / H);
            Vec3  dir = normalize(Vec3(u, v, -1.5f));

            Vec3 color = rayColorIterative(cameraPos, dir, NUM_BOUNCES);

            // Store the color as 3 bytes (clamp to [0,1] first).
            int idx = (py * W + px) * 3;
            image[idx + 0] = (unsigned char)(std::fmin(color.x, 1.0f) * 255);
            image[idx + 1] = (unsigned char)(std::fmin(color.y, 1.0f) * 255);
            image[idx + 2] = (unsigned char)(std::fmin(color.z, 1.0f) * 255);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("cpu: %.1f ms\n", ms);
    
    // Write a PPM image file (a very simple image format).
    const char* outPath = "../images/output/cpu_bvh.ppm";
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
