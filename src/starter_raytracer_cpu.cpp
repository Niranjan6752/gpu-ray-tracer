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

// ---- A sphere: a center, a radius, and a color ----------------------------
struct Sphere {
    Vec3  center;
    float radius;
    Vec3  color;
    bool  isFloor;   // if true, we draw a checkerboard instead of a solid color
};

// The whole scene is just a list of spheres. (The floor is a giant sphere.)
std::vector<Sphere> scene = {
    { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), true  }, // floor
    { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), false }, // red
    { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), false }, // green
    { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), false }, // blue
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

// Find the closest sphere the ray hits. Returns its index (or -1) and the distance.
int closestHit(Vec3 origin, Vec3 dir, float& tHit) {
    int   best = -1;
    float closest = 1e30f;
    for (int i = 0; i < (int)scene.size(); i++) {
        float t = hitSphere(scene[i], origin, dir);
        if (t > 0 && t < closest) { closest = t; best = i; }
    }
    tHit = closest;
    return best;
}

// Compute the color seen along one ray.
Vec3 rayColor(Vec3 origin, Vec3 dir) {
    // TRY IT: move the light around and see how shading/shadows change.
    // x = right, y = up, z = toward the camera.
    Vec3  lightPos(4, 6, 2);
    float tHit;
    int   hit = closestHit(origin, dir, tHit);

    if (hit < 0) {
        // Miss: draw a simple sky gradient (white near horizon -> blue up top).
        float t = 0.5f * (dir.y + 1.0f);
        return Vec3(1, 1, 1) * (1 - t) + Vec3(0.4f, 0.6f, 1.0f) * t;
    }

    // The point we hit, and the surface normal there.
    Vec3 p = origin + dir * tHit;
    Vec3 n = normalize(p - scene[hit].center);

    // Base color: checkerboard for the floor, solid color otherwise.
    Vec3 base = scene[hit].color;
    if (scene[hit].isFloor) {
        int cx = (int)std::floor(p.x), cz = (int)std::floor(p.z);
        bool white = ((cx + cz) & 1) == 0;
        base = white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f);
    }

    // Diffuse lighting: how directly does the surface face the light?
    Vec3  toLight = normalize(lightPos - p);
    float diffuse = std::fmax(0.0f, dot(n, toLight));

    // Hard shadow: shoot a ray toward the light; if it hits something, we're shadowed.
    float shadowT;
    int   blocked = closestHit(p + n * 0.001f, toLight, shadowT);
    float shade = (blocked >= 0) ? 0.3f : 1.0f;

    float brightness = 0.15f + diffuse * shade;   // 0.15 = a little ambient light
    return base * brightness;
}

int main() {
    const int W = 800, H = 600;
    std::vector<unsigned char> image(W * H * 3);

    // TRY IT: move the camera and see how the view changes.
    // x = right, y = up, z = toward the viewer (spheres sit near z = -1).
    Vec3  cameraPos(0, 1, 3);
    float aspect = (float)W / H;

    // For every pixel, build a ray from the camera and compute its color.
    for (int py = 0; py < H; py++) {
        for (int px = 0; px < W; px++) {
            // Map pixel (px,py) to a direction through the image plane.
            float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
            float v = (1.0f - 2.0f * (py + 0.5f) / H);
            Vec3  dir = normalize(Vec3(u, v, -1.5f));

            Vec3 color = rayColor(cameraPos, dir);

            // Store the color as 3 bytes (clamp to [0,1] first).
            int idx = (py * W + px) * 3;
            image[idx + 0] = (unsigned char)(std::fmin(color.x, 1.0f) * 255);
            image[idx + 1] = (unsigned char)(std::fmin(color.y, 1.0f) * 255);
            image[idx + 2] = (unsigned char)(std::fmin(color.z, 1.0f) * 255);
        }
    }

    // Write a PPM image file (a very simple image format).
    FILE* f = fopen("image.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image.data(), 1, image.size(), f);
    fclose(f);
    printf("Wrote image.ppm (%dx%d)\n", W, H);
    return 0;
}
