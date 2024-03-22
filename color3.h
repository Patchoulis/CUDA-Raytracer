#include <cstdint>
#include "Vec3.h"

#pragma once

class Color3 {
    public:
        uint8_t r, g, b;
        __host__ __device__ Color3(uint8_t r, uint8_t g, uint8_t b);
        __host__ __device__ Color3(uint8_t r, uint8_t g, uint8_t b, float intensity);
        __host__ __device__ Color3 getColor(const float intensity);
        __host__ __device__ void setColor(uint8_t r, uint8_t g, uint8_t b);
        __host__ __device__ uint32_t toUint32() const;
        __host__ __device__ uint32_t toUint32(float intensity) const;
        __host__ __device__ uint32_t toUint32(const Vec3& intensity) const;
        __host__ __device__ Color3 operator+(const Color3& other) const;
        __host__ __device__ Color3& operator+=(const Color3& other);
        __host__ __device__ Color3 operator*(const Vec3& other) const;
        __host__ __device__ Color3& operator*=(const Vec3& other);
        __host__ __device__ Color3 operator*(const float& other) const;
        __host__ __device__ Color3& operator*=(const float& other);
};