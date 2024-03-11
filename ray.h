#include "Vec3.h"

#pragma once

class Ray {
    public:
        Vec3 Pos,Dir;
        __host__ __device__ Ray(Vec3 Pos,Vec3 Dir);
        __host__ __device__ Ray operator*(const float other) const;
        __host__ __device__ Ray& operator*=(const float other);
        __host__ __device__ Vec3 getDirection() const;
        __host__ __device__ Vec3 getPos() const;
        __host__ __device__ Vec3 getPosAtDist(float dist) const;
};