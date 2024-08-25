#include "sphere.h"
#include <cmath>
#include <iostream>

__host__ __device__ Sphere::Sphere(Vec3 Pos, Material material) : Pos(Pos), material(material) {}

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

__host__ __device__ RayIntersectResult Triangle::rayIntersect(Ray ray) {
    Vec3 borderOne = this->p2 - this->p1;
    Vec3 borderTwo = this->p3 - this->p1;
    Vec3 pvec = ray.getDirection().cross(borderTwo);
    float det = borderOne.dot(pvec);
    if (det < EPSILON){
        return RayIntersectResult{false, 0,0,0};
    }
    float invDet = 1/det;
    Vec3 tvec = ray.getPos() - this->p1;
    float u = tvec.dot(pvec) * invDet;
    Vec3 qvec = tvec.cross(borderOne);
    float v = ray.getDirection().dot(qvec) * invDet;
    if (u < 0 || u > 1 || v < 0 || u+v > 1) {
        return RayIntersectResult{false, 0,0,0};
    }
    float t = borderTwo.dot(qvec) * invDet;
    return RayIntersectResult{true, t,u,v};
}