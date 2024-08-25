#include <iostream>
#include <cstdint>

#pragma once

class Vec2 {
    private:
        float x, y;
    public:
        __host__ __device__ Vec2(float x,float y);
        __host__ __device__ Vec2 operator+(const Vec2& other) const;
        __host__ __device__ Vec2 operator+(const float other) const;
        __host__ __device__ Vec2 operator-(const Vec2& other) const;
        __host__ __device__ Vec2 operator-(const float other) const;
        __host__ __device__ Vec2 operator*(const float other) const;
        __host__ __device__ Vec2 operator*(const Vec2& other) const;
        __host__ __device__ Vec2 operator/(const float other) const;
        __host__ __device__ Vec2 operator/(const Vec2& other) const;
        __host__ __device__ Vec2 operator-() const;

        __host__ __device__ Vec2& operator+=(const Vec2& other);
        __host__ __device__ Vec2& operator+=(const float other);
        __host__ __device__ Vec2& operator-=(const Vec2& other);
        __host__ __device__ Vec2& operator-=(const float other);
        __host__ __device__ Vec2& operator*=(const float other);
        __host__ __device__ Vec2& operator*=(const Vec2& other);

        __host__ __device__ float& operator[](const uint& other);
        __host__ __device__ const float& operator[](const uint& other) const;

        __host__ __device__ const float& getX() const;
        __host__ __device__ const float& getY() const;
        __host__ __device__ const float& getZ() const;

        __host__ __device__ void setX(float x);
        __host__ __device__ void setY(float y);
        __host__ __device__ void setZ(float z);

        __host__ __device__ float dot(const Vec2& other) const;
        __host__ __device__ Vec2 cross(const Vec2& other) const;
        __host__ __device__ uint32_t toUint32() const;

        friend std::ostream& operator<<(std::ostream& os, const Vec2& vec);
};