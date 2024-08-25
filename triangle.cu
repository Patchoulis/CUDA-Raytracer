#include "triangle.h"
#include <cmath>
#include <iostream>

__host__ __device__ Triangle::Triangle(Vec3 p1,Vec3 p2,Vec3 p3, Material material) : p1(p1), p2(p2), p3(p3), material(material), norm(this->calcNorm()) {}

__host__ __device__ Triangle::Triangle(Vec3 p1,Vec3 p2,Vec3 p3,Vec3 norm, Material material) : p1(p1), p2(p2), p3(p3), material(material), norm(norm) {}

__host__ __device__ Vec3 Triangle::calcNorm() const {
    Vec3 borderOne = p3 - p1;
    Vec3 borderTwo = p3 - p2;
    return borderOne.cross(borderTwo).unitVector();
}

__host__ __device__ Vec3 Triangle::getCentroid() const {
    return (this->p1 + this->p2 + this->p3)/3.0f;
}

__host__ __device__ float Triangle::getMinX() const {
    return min(this->p3.x,min(this->p1.x,this->p2.x));
}

__host__ __device__ float Triangle::getMinY() const {
    return min(this->p3.y,min(this->p1.y,this->p2.y));
}

__host__ __device__ float Triangle::getMinZ() const {
    return min(this->p3.z,min(this->p1.z,this->p2.z));
}

__host__ __device__ float Triangle::getMaxX() const {
    return max(this->p3.x,max(this->p1.x,this->p2.x));
}

__host__ __device__ float Triangle::getMaxY() const {
    return max(this->p3.y,max(this->p1.y,this->p2.y));
}

__host__ __device__ float Triangle::getMaxZ() const {
    return max(this->p3.z,max(this->p1.z,this->p2.z));
}

__host__ __device__ const Material& Triangle::getMaterial() const {
    return this->material;
}