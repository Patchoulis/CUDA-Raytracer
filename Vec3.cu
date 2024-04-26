#include "Vec3.h"
#include <cmath>

__host__ __device__ Vec3::Vec3(float x,float y,float z) : x(x), y(y), z(z){}

__host__ __device__ Vec3 Vec3::unitVector() const {
    float magn = this->magnitude();
    return Vec3(this->x/magn, this->y/magn, this->z/magn);
}

__host__ __device__ float Vec3::magnitude() const {
    return sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
}

__host__ __device__ Vec3 Vec3::operator+(const Vec3& other) const {
    return Vec3(this->x + other.getX(), this->y + other.getY(), this->z + other.getZ());
}

__host__ __device__ Vec3 Vec3::operator+(const float other) const {
    return Vec3(this->x + other, this->y + other, this->z + other);
}

__host__ __device__ Vec3 Vec3::operator-(const Vec3& other) const {
    return Vec3(this->x - other.getX(), this->y - other.getY(), this->z - other.getZ());
}

__host__ __device__ Vec3 Vec3::operator-(const float other) const {
    return Vec3(this->x - other, this->y - other, this->z - other);
}

__host__ __device__ Vec3 Vec3::operator*(const float other) const {
    return Vec3(this->x * other, this->y * other, this->z * other);
}

__host__ __device__ Vec3 Vec3::operator*(const Vec3& other) const {
    return Vec3(this->x * other.getX(), this->y * other.getY(), this->z * other.getZ());
}

__host__ __device__ Vec3 Vec3::operator/(const float other) const {
    if (other == 0) {
        return *this;
    }
    return Vec3(this->x / other, this->y / other, this->z / other);
}

__host__ __device__ Vec3 Vec3::operator/(const Vec3& other) const {
    if (other.getZ() == 0 || other.getZ() == 0 || other.getZ() == 0) {
        return *this;
    }
    return Vec3(this->x / other.getX(), this->y / other.getY(), this->z / other.getZ());
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
    this->x+=other.getX();
    this->y+=other.getY();
    this->z+=other.getZ();
    return *this;
}

__host__ __device__ Vec3& Vec3::operator-=(const float other) {
    this->x-=other;
    this->y-=other;
    this->z-=other;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator-=(const Vec3& other) {
    this->x-=other.getX();
    this->y-=other.getY();
    this->z-=other.getZ();
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const float other) {
    this->x*=other;
    this->y*=other;
    this->z*=other;
    return *this;
}

__host__ __device__ Vec3& Vec3::operator*=(const Vec3& other) {
    this->x*=other.getX();
    this->y*=other.getY();
    this->z*=other.getZ();
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Vec3& vec) {
    os << "(" << vec.getX() << ", " << vec.getY() << ", " << vec.getZ() << ")";
    return os;
}

__host__ __device__ float Vec3::dot(const Vec3& other) const {
    return other.getX()*this->x + other.getY()*this->y + other.getZ()*this->z;
}

__host__ __device__ Vec3 Vec3::cross(const Vec3& other) const {
    return Vec3(this->y*other.getZ() - this->z*other.getY(), this->z*other.getX()-this->x*other.getZ(),this->x*other.getY()-this->y*other.getX());
}

__host__ __device__ const float& Vec3::getX() const {
    return this->x;
}

__host__ __device__ const float& Vec3::getY() const {
    return this->y;
}

__host__ __device__ const float& Vec3::getZ() const {
    return this->z;
}

__host__ __device__ void Vec3::setX(float x) {
    this->x = x;
}

__host__ __device__ void Vec3::setY(float y) {
    this->y = y;
}

__host__ __device__ void Vec3::setZ(float z) {
    this->z = z;
}

__host__ __device__ uint32_t Vec3::toUint32() const {
    uint32_t x = static_cast<uint8_t>(max(0.0f,min(255.0f,this->x)));
    uint32_t y = static_cast<uint8_t>(max(0.0f,min(255.0f,this->y)));
    uint32_t z = static_cast<uint8_t>(max(0.0f,min(255.0f,this->z)));
    return (x << 24) | (y << 16) | (z << 8) | static_cast<uint32_t>(255);
}