#include "color3.h"

#pragma once

class Material {
    public:
        Color3 color;
        Vec3 Albedo;
        Vec3 Emissivity;
        Vec3 Reflectivity;
        float Roughness;
        Material(Color3 color, Vec3 Albedo = Vec3(0,0,0), Vec3 Emissivity = Vec3(0,0,0), Vec3 Reflectivity = Vec3(0,0,0), float Roughness = 0);
        __host__ __device__ void setColor(uint8_t r, uint8_t g, uint8_t b);
        __host__ __device__ void setColor(Color3 color);
        __host__ __device__ const Color3& getColor() const;
};