#include <iostream>

#pragma once

class Vec3 {
    private:
        float x, y, z;
    public:
        __host__ __device__ Vec3(float x,float y,float z);
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

        __host__ __device__ float getX() const;
        __host__ __device__ float getY() const;
        __host__ __device__ float getZ() const;

        __host__ __device__ void setX(float x);
        __host__ __device__ void setY(float y);
        __host__ __device__ void setZ(float z);

        __host__ __device__ float dot(const Vec3& other) const;
        __host__ __device__ Vec3 cross(const Vec3& other) const;

        friend std::ostream& operator<<(std::ostream& os, const Vec3& vec);
};