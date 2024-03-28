#include "material.h"

Material::Material(Color3 color, Vec3 Albedo = Vec3(0,0,0), Vec3 Emissivity = Vec3(0,0,0), Vec3 Reflectivity = Vec3(0,0,0), float Roughness = 0) : color(color), Albedo(Albedo), Emissivity(Emissivity),
     Reflectivity(Reflectivity), Roughness(Roughness) {}

__host__ __device__ const Color3& Material::getColor() const {
    return this->color;
}

__host__ __device__ void Material::setColor(uint8_t r, uint8_t g, uint8_t b) {
    this->color.setColor(r,g,b);
}

__host__ __device__ void Material::setColor(Color3 color) {
    this->color = color;
}