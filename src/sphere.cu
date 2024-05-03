#include "sphere.h"
#include <cmath>
#include <iostream>
// #include <triangle.h>

__host__ __device__ Sphere::Sphere(Vec3 Pos, Material material) : Pos(Pos), material(material) {}

/* Duplicate implementation in triangle.cu
__host__ __device__ Vec3 Triangle::calcNorm(Vec3 Center) const {
    Vec3 borderOne = this->p3 - this->p1;
    Vec3 borderTwo = this->p3 - this->p2;

    Vec3 NormDir = (this->p3 + this->p2 + this->p1)/3.0f - Center;
    Vec3 Norm = borderOne.cross(borderTwo).unitVector();
    if (NormDir.dot(Norm) < 0) {
        return -Norm;
    }
    return Norm;
}

__host__ __device__ const Vec3& Triangle::getNorm() const {
    return this->norm;
}

__host__ __device__ void Triangle::updatePos(Vec3& Diff) {
    this->p1 += Diff;
    this->p2 += Diff;
    this->p3 += Diff;
}
*/

/* Not sure why this is here when there was no declaration
__host__ __device__ const Color3& Triangle::getColor() const {
    return this->material.getColor();
}

__host__ __device__ void Triangle::setColor(uint8_t r, uint8_t g, uint8_t b) {
    this->material.setColor(r,g,b);
}

__host__ __device__ void Triangle::setColor(Color3 color) {
    this->material.setColor(color);
}

__host__ __device__ const Material& Triangle::getMaterial() const {
    return this->material;
}
*/


