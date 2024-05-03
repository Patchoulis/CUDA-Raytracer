#include "triangle.h"
#include <cmath>
#include <iostream>
#include <sphere.h>

__host__ __device__ Triangle::Triangle(Vec3 p1,Vec3 p2,Vec3 p3, Material material) : lp1(p1),lp2(p2),lp3(p3), p1(p1), p2(p2), p3(p3), centroid(this->calcCentroid()), material(material), norm(this->calcNorm()) {}

__host__ __device__ Triangle::Triangle(Vec3 lp1,Vec3 lp2,Vec3 lp3, Quaternion CFrame, Material material) : lp1(lp1),lp2(lp2),lp3(lp3),
 p1(CFrame.getPos()+CFrame.getRightVector()*lp1.getX()+CFrame.getUpVector()*lp1.getY()-CFrame.getLookVector()*lp1.getZ()),
 p2(CFrame.getPos()+CFrame.getRightVector()*lp2.getX()+CFrame.getUpVector()*lp2.getY()-CFrame.getLookVector()*lp2.getZ()),
 p3(CFrame.getPos()+CFrame.getRightVector()*lp3.getX()+CFrame.getUpVector()*lp3.getY()-CFrame.getLookVector()*lp3.getZ()), centroid(this->calcCentroid()), material(material), norm(this->calcNorm()) {}

__host__ __device__ Vec3 Triangle::calcNorm() const {
    Vec3 borderOne = p3 - p1;
    Vec3 borderTwo = p3 - p2;
    return borderOne.cross(borderTwo).unitVector();
}

__host__ __device__ Vec3 Triangle::calcNorm(Vec3 Center) const {
    Vec3 borderOne = this->p3 - this->p1;
    Vec3 borderTwo = this->p3 - this->p2;

    Vec3 NormDir = (this->p3 + this->p2 + this->p1)/3.0f - Center;
    Vec3 Norm = borderOne.cross(borderTwo).unitVector();
    //std::cout << this->p1 << this->p2 << this->p3 << NormDir.dot(Norm) << Center << NormDir << " " << Norm << "\n";
    if (NormDir.dot(Norm) < 0) {
        return -Norm;
    }
    return Norm;
}

__host__ __device__ const Vec3& Triangle::getNorm() const {
    return this->norm;
}

__host__ __device__ Vec3 Triangle::calcCentroid() const {
    return (this->p1 + this->p2 + this->p3)/3;
}

__host__ __device__ const Vec3& Triangle::getCentroid() const {
    return this->centroid;
}

__host__ __device__ void Triangle::updatePos(Vec3& Diff) {
    this->p1 += Diff;
    this->p2 += Diff;
    this->p3 += Diff;
}

__host__ __device__ float Triangle::getMinX() const {
    return min(this->p3.getX(),min(this->p1.getX(),this->p2.getX()));
}

__host__ __device__ float Triangle::getMinY() const {
    return min(this->p3.getY(),min(this->p1.getY(),this->p2.getY()));
}

__host__ __device__ float Triangle::getMinZ() const {
    return min(this->p3.getZ(),min(this->p1.getZ(),this->p2.getZ()));
}

__host__ __device__ float Triangle::getMaxX() const {
    return max(this->p3.getX(),max(this->p1.getX(),this->p2.getX()));
}

__host__ __device__ float Triangle::getMaxY() const {
    return max(this->p3.getY(),max(this->p1.getY(),this->p2.getY()));
}

__host__ __device__ float Triangle::getMaxZ() const {
    return max(this->p3.getZ(),max(this->p1.getZ(),this->p2.getZ()));
}

__host__ __device__ const Material& Triangle::getMaterial() const {
    return this->material;
}

__host__ __device__ const Vec3& Triangle::getGlobalV1() const {
    return this->p1;
}

__host__ __device__ const Vec3& Triangle::getGlobalV2() const {
    return this->p2;
}

__host__ __device__ const Vec3& Triangle::getGlobalV3() const {
    return this->p3;
}
// __host__ __device__ RayIntersectResult Triangle::rayIntersect(Ray ray) {
//     Vec3 borderOne = this->p2 - this->p1;
//     Vec3 borderTwo = this->p3 - this->p1;
//     Vec3 pvec = ray.getDirection().cross(borderTwo);
//     float det = borderOne.dot(pvec);
//     if (det < EPSILON){
//         return RayIntersectResult{false, 0,0,0};
//     }
//     float invDet = 1/det;
//     Vec3 tvec = ray.getPos() - this->p1;
//     float u = tvec.dot(pvec) * invDet;
//     Vec3 qvec = tvec.cross(borderOne);
//     float v = ray.getDirection().dot(qvec) * invDet;
//     if (u < 0 || u > 1 || v < 0 || u+v > 1) {
//         return RayIntersectResult{false, 0,0,0};
//     }
//     float t = borderTwo.dot(qvec) * invDet;
//     return RayIntersectResult{true, t,u,v};
// }
