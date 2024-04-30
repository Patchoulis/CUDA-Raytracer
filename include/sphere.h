#pragma once

#include "Vec3.h"
#include "ray.h"
#include "material.h"
#include "quaternion.h"
#include <utility>

#define EPSILON 0.00001

struct RayIntersectResult {
    bool hit;
    float t;
    float u;
    float v;
};

class Sphere {
    private:
        Vec3 Pos;
        Material material;
    public:
        __host__ __device__ Sphere(Vec3 Pos, Material material = Material(Color3(200,200,200)));
        __host__ __device__ RayIntersectResult rayIntersect(Ray ray);
        __host__ __device__ const Vec3 getNorm() const;
        __host__ __device__ void setColor(uint8_t r, uint8_t g, uint8_t b);
        __host__ __device__ void setColor(Color3 color);
        __host__ __device__ const Color3& getColor() const;
        __host__ __device__ const Material& getMaterial() const;
        __host__ __device__ void updatePos(Vec3& Diff);
};