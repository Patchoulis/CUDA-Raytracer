#pragma once

#include "Vec3.h"

class Quaternion {
private:
    Vec3 Pos;
    Vec3 Up;
    Vec3 Right;
    Vec3 Look;
public:
    __host__ __device__ Quaternion(Vec3 Pos = Vec3(0.0f,0.0f,0.0f), Vec3 Up = Vec3(0.0f,1.0f,0.0f),  Vec3 Right = Vec3(1.0f,0.0f,0.0f), Vec3 Look = Vec3(0.0f,0.0f,-1.0f));
    __host__ __device__ const Vec3& getPos() const;
    __host__ __device__ const Vec3& getLookVector() const;
    __host__ __device__ const Vec3& getUpVector() const;
    __host__ __device__ const Vec3& getRightVector() const;

    __host__ __device__ void setPos(Vec3 other);
    __host__ __device__ void setLookVector(Vec3 other);
    __host__ __device__ void setUpVector(Vec3 other);
    __host__ __device__ void setRightVector(Vec3 other);

    __host__ __device__ Quaternion operator+(const Vec3& other) const;
    __host__ __device__ Quaternion& operator+=(const Vec3& other);

    __host__ __device__ Quaternion& rotate(const Vec3& other, double deg);

    friend std::ostream& operator<<(std::ostream& os, const Quaternion& vec);
};