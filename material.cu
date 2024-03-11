#include "material.h"

Material::Material(Color3 color, float specularity, float Refractivity, float Reflectivity, float Transparency) : color(color), Specularity(specularity), Refractivity(Refractivity),
     Reflectivity(Reflectivity), Transparency(Transparency) {}

__host__ __device__ const Color3& Material::getColor() const {
    return this->color;
}

__host__ __device__ void Material::setColor(uint8_t r, uint8_t g, uint8_t b) {
    this->color.setColor(r,g,b);
}

__host__ __device__ void Material::setColor(Color3 color) {
    this->color = color;
}