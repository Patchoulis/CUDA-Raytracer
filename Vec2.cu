#include "Vec2.h"
#include <cmath>

__host__ __device__ Vec2::Vec2(float x,float y) : x(x), y(y) {}

__host__ __device__ Vec2 Vec2::operator+(const Vec2& other) const {
    return Vec2(this->x + other.getX(), this->y + other.getY());
}

__host__ __device__ Vec2 Vec2::operator+(const float other) const {
    return Vec2(this->x + other, this->y + other);
}

__host__ __device__ Vec2 Vec2::operator-(const Vec2& other) const {
    return Vec2(this->x - other.getX(), this->y - other.getY());
}

__host__ __device__ Vec2 Vec2::operator-(const float other) const {
    return Vec2(this->x - other, this->y - other);
}

__host__ __device__ Vec2 Vec2::operator*(const float other) const {
    return Vec2(this->x * other, this->y * other);
}

__host__ __device__ Vec2 Vec2::operator*(const Vec2& other) const {
    return Vec2(this->x * other.getX(), this->y * other.getY());
}

__host__ __device__ Vec2 Vec2::operator/(const float other) const {
    if (other == 0) {
        return *this;
    }
    return Vec2(this->x / other, this->y / other);
}

__host__ __device__ Vec2 Vec2::operator/(const Vec2& other) const {
    if (other.getX() == 0 || other.getY() == 0) {
        return *this;
    }
    return Vec2(this->x / other.getX(), this->y / other.getY());
}

__host__ __device__ Vec2 Vec2::operator-() const {
    return Vec2(-this->x, -this->y);
}

__host__ __device__ float& Vec2::operator[](const uint& other) {
    if (other == 0) {
        return this->x;
    }
    return this->y;
}

__host__ __device__ const float& Vec2::operator[](const uint& other) const {
    if (other == 0) {
        return this->x;
    }
    return this->y;
}

__host__ __device__ Vec2& Vec2::operator+=(const float other) {
    this->x+=other;
    this->y+=other;
    return *this;
}

__host__ __device__ Vec2& Vec2::operator+=(const Vec2& other) {
    this->x+=other.getX();
    this->y+=other.getY();
    return *this;
}

__host__ __device__ Vec2& Vec2::operator-=(const float other) {
    this->x-=other;
    this->y-=other;
    return *this;
}

__host__ __device__ Vec2& Vec2::operator-=(const Vec2& other) {
    this->x-=other.getX();
    this->y-=other.getY();
    return *this;
}

__host__ __device__ Vec2& Vec2::operator*=(const float other) {
    this->x*=other;
    this->y*=other;
    return *this;
}

__host__ __device__ Vec2& Vec2::operator*=(const Vec2& other) {
    this->x*=other.getX();
    this->y*=other.getY();
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Vec2& vec) {
    os << "(" << vec.getX() << ", " << vec.getY() << ")";
    return os;
}

__host__ __device__ const float& Vec2::getX() const {
    return this->x;
}

__host__ __device__ const float& Vec2::getY() const {
    return this->y;
}

__host__ __device__ void Vec2::setX(float x) {
    this->x = x;
}

__host__ __device__ void Vec2::setY(float y) {
    this->y = y;
}