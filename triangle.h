#include "Vec3.h"
#include "ray.h"
#include "material.h"
#include "quaternion.h"
#include <utility>

#pragma once

class Triangle {
    private:
        __host__ __device__ Vec3 calcNorm() const;
    public:
        Vec3 p1,p2,p3;
        Vec3 norm;
        Material material;
        __host__ __device__ Triangle(Vec3 p1,Vec3 p2,Vec3 p3, Material material = Material(Vec3(200,200,200)));
        __host__ __device__ Triangle(Vec3 p1,Vec3 p2,Vec3 p3,Vec3 norm, Material material = Material(Vec3(200,200,200)));
        __host__ __device__ const Material& getMaterial() const;
        __host__ __device__ float getMinX() const;
        __host__ __device__ float getMinY() const;
        __host__ __device__ float getMinZ() const;
        __host__ __device__ float getMaxX() const;
        __host__ __device__ float getMaxY() const;
        __host__ __device__ float getMaxZ() const;
        
        __host__ __device__ Vec3 getCentroid() const;
};