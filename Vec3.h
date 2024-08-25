#include <iostream>
#include <cstdint>

#pragma once

class alignas(16) Vec3 {
    public:
        float x, y, z;
        __host__ __device__ Vec3(float x=0,float y=0,float z=0);
        __host__ __device__ Vec3 unitVector() const;
        __host__ __device__ float magnitude() const;
        __host__ __device__ Vec3 operator+(const Vec3& other) const;
        __host__ __device__ Vec3 operator+(const float other) const;
        __host__ __device__ Vec3 operator-(const Vec3& other) const;
        __host__ __device__ Vec3 operator-(const float other) const;
        __host__ __device__ Vec3 operator*(const float other) const;
        __host__ __device__ Vec3 operator*(const Vec3& other) const;
        __host__ __device__ Vec3 operator/(const float other) const;
        __host__ __device__ Vec3 operator/(const Vec3& other) const;
        __host__ __device__ Vec3 operator-() const;

        __host__ __device__ Vec3& operator+=(const Vec3& other);
        __host__ __device__ Vec3& operator+=(const float other);
        __host__ __device__ Vec3& operator-=(const Vec3& other);
        __host__ __device__ Vec3& operator-=(const float other);
        __host__ __device__ Vec3& operator*=(const float other);
        __host__ __device__ Vec3& operator*=(const Vec3& other);

        __host__ __device__ float& operator[](const uint& other);
        __host__ __device__ const float& operator[](const uint& other) const;

        __host__ __device__ float dot(const Vec3& other) const;
        __host__ __device__ Vec3 cross(const Vec3& other) const;
        __host__ __device__ uint32_t toUint32() const;

        friend std::ostream& operator<<(std::ostream& os, const Vec3& vec);
        friend __device__ void atomicAddVec3(Vec3& address, const Vec3& val);
};