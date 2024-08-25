#include "Vec3.h"
#include <cmath>

__host__ __device__ Vec3::Vec3(float x,float y,float z) : x(x), y(y), z(z){}

__host__ __device__ Vec3 Vec3::unitVector() const {
    float magnInv = 1/this->magnitude();
    return Vec3(this->x * magnInv, this->y * magnInv, this->z * magnInv);
}

__host__ __device__ float Vec3::magnitude() const {
    return sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
}

__host__ __device__ Vec3 Vec3::operator+(const Vec3& other) const {
    return Vec3(this->x + other.x, this->y + other.y, this->z + other.z);
}

__host__ __device__ Vec3 Vec3::operator+(const float other) const {
    return Vec3(this->x + other, this->y + other, this->z + other);
}

__host__ __device__ Vec3 Vec3::operator-(const Vec3& other) const {
    return Vec3(this->x - other.x, this->y - other.y, this->z - other.z);
}

__host__ __device__ Vec3 Vec3::operator-(const float other) const {
    return Vec3(this->x - other, this->y - other, this->z - other);
}

__host__ __device__ Vec3 Vec3::operator*(const float other) const {
    return Vec3(this->x * other, this->y * other, this->z * other);
}

__host__ __device__ Vec3 Vec3::operator*(const Vec3& other) const {
    return Vec3(this->x * other.x, this->y * other.y, this->z * other.z);
}

__host__ __device__ Vec3 Vec3::operator/(const float other) const {
    if (other == 0) {
        return *this;
    }
    return Vec3(this->x / other, this->y / other, this->z / other);
}

__host__ __device__ Vec3 Vec3::operator/(const Vec3& other) const {
    if (other.x == 0 || other.y == 0 || other.z == 0) {
        return *this;
    }
    return Vec3(this->x / other.x, this->y / other.y, this->z / other.z);
}

__host__ __device__ Vec3 Vec3::operator-() const {
    return Vec3(-this->x, -this->y, -this->z);
}

__host__ __device__ float& Vec3::operator[](const uint& other) {
    if (other == 0) {
        return this->x;
    }
    if (other == 1) {
        return this->y;
    }
    return this->z;
}

__host__ __device__ const float& Vec3::operator[](const uint& other) const {
    if (other == 0) {
        return this->x;
    }
    if (other == 1) {
        return this->y;
    }
    return this->z;
}

__host__ __device__ Vec3& Vec3::operator+=(const float other) {
    this->x+=other;
    this->y+=other;
    this->z+=other;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator+=(const Vec3& other) {
    this->x+=other.x;
    this->y+=other.y;
    this->z+=other.z;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator-=(const float other) {
    this->x-=other;
    this->y-=other;
    this->z-=other;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator-=(const Vec3& other) {
    this->x-=other.x;
    this->y-=other.y;
    this->z-=other.z;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const float other) {
    this->x*=other;
    this->y*=other;
    this->z*=other;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const Vec3& other) {
    this->x*=other.x;
    this->y*=other.y;
    this->z*=other.z;
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

__host__ __device__ float Vec3::dot(const Vec3& other) const {
    return other.x*this->x + other.y*this->y + other.z*this->z;
}

__host__ __device__ Vec3 Vec3::cross(const Vec3& other) const {
    return Vec3(this->y*other.z - this->z*other.y, this->z*other.x-this->x*other.z,this->x*other.y-this->y*other.x);
}

__host__ __device__ uint32_t Vec3::toUint32() const {
    uint32_t x = static_cast<uint8_t>(max(0.0f,min(255.0f,this->x)));
    uint32_t y = static_cast<uint8_t>(max(0.0f,min(255.0f,this->y)));
    uint32_t z = static_cast<uint8_t>(max(0.0f,min(255.0f,this->z)));
    return (x << 24) | (y << 16) | (z << 8) | static_cast<uint32_t>(255);
}

__device__ void atomicAddVec3(Vec3& address, const Vec3& val) {
    atomicAdd(&(address.x), val.x);
    atomicAdd(&(address.y), val.y);
    atomicAdd(&(address.z), val.z);
}