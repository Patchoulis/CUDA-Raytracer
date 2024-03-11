#include "color3.h"

__host__ __device__ Color3::Color3(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}

__host__ __device__ Color3::Color3(uint8_t r, uint8_t g, uint8_t b, float intensity) : r(r*intensity), g(g*intensity), b(b*intensity) {}

__host__ __device__ Color3 Color3::getColor(float intensity) {
    return Color3(r*intensity,g*intensity,b*intensity);
}

__host__ __device__ void Color3::setColor(uint8_t r, uint8_t g, uint8_t b) {
    this->r = r;
    this->b = b;
    this->g = g;
}

__host__ __device__ uint32_t Color3::toUint32(float intensity) const {
    return (static_cast<uint32_t>(intensity*this->r) << 24) | (static_cast<uint32_t>(intensity*this->g) << 16) | (static_cast<uint32_t>(intensity*this->b) << 8) | static_cast<uint32_t>(255);
}

__host__ __device__ Color3 Color3::operator+(const Color3& other) const {
    uint8_t newR = other.r + this->r;
    uint8_t newG = other.g + this->g;
    uint8_t newB = other.b + this->b;
    // Overflow detection
    return Color3((newR >= this->r) ? newR : 255, (newG >= this->g) ? newG : 255, (newB >= this->b) ? newB : 255);
}