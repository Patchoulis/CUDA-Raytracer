#pragma once
#include "Vec3.h"

struct RayIntersectResult {
    bool hit;
    float t;
    float u;
    float v;
};

class Ray {
    public:
        Vec3 Pos,Dir;
        RayIntersectResult hit;
        __host__ __device__ Ray(Vec3 Pos,Vec3 Dir);
        __host__ __device__ Ray operator*(const float other) const;
        __host__ __device__ Ray& operator*=(const float other);
        __host__ __device__ const Vec3& getDirection() const;
        __host__ __device__ const Vec3& getPos() const;
        __host__ __device__ Vec3 getPosAtDist(float dist) const;
        __host__ __device__ bool hasHit() const;
};