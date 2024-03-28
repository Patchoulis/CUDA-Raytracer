#include "Vec3.h"
#include "ray.h"
#include "material.h"
#include "quaternion.h"
#include <utility>

#pragma once

class Triangle {
    private:
        Vec3 lp1,lp2,lp3;
        Vec3 p1,p2,p3, centroid;
        Material material;
        Vec3 norm;
        __host__ __device__ Vec3 calcNorm() const;
        __host__ __device__ Vec3 calcNorm(Vec3 Center) const;
        __host__ __device__ Vec3 calcCentroid() const;
    public:
        __host__ __device__ Triangle(Vec3 p1,Vec3 p2,Vec3 p3, Material material = Material(Color3(200,200,200)));
        __host__ __device__ Triangle(Vec3 p1,Vec3 p2,Vec3 p3, Quaternion CFrame, Material material = Material(Color3(200,200,200)));
        __host__ __device__ const Vec3& getNorm() const;
        __host__ __device__ void setColor(uint8_t r, uint8_t g, uint8_t b);
        __host__ __device__ void setColor(Color3 color);
        __host__ __device__ const Color3& getColor() const;
        __host__ __device__ const Material& getMaterial() const;
        __host__ __device__ void updatePos(Vec3& Diff);
        __host__ __device__ float getMinX() const;
        __host__ __device__ float getMinY() const;
        __host__ __device__ float getMinZ() const;
        __host__ __device__ float getMaxX() const;
        __host__ __device__ float getMaxY() const;
        __host__ __device__ float getMaxZ() const;

        __host__ __device__ const Vec3& getGlobalV1() const;
        __host__ __device__ const Vec3& getGlobalV2() const;
        __host__ __device__ const Vec3& getGlobalV3() const;

        __host__ __device__ const Vec3& getCentroid() const;
};