#include "quaternion.h"
#include <cmath>

__host__ __device__ Quaternion::Quaternion(Vec3 Pos, Vec3 Up, Vec3 Right, Vec3 Look) : Pos(Pos), Up(Up), Right(Right), Look(Look) {}

__host__ __device__ Vec3 Quaternion::getPos() const{
    return this->Pos;
}

__host__ __device__ Vec3 Quaternion::getRightVector() const{
    return this->Right;
}

__host__ __device__ Vec3 Quaternion::getUpVector() const{
    return this->Up;
}

__host__ __device__ Vec3 Quaternion::getLookVector() const{
    return this->Look;
}

__host__ __device__ void Quaternion::setPos(Vec3 other) {
    this->Pos = other;
}

__host__ __device__ void Quaternion::setRightVector(Vec3 other) {
    this->Right = other;
}

__host__ __device__ void Quaternion::setUpVector(Vec3 other) {
    this->Up = other;
}

__host__ __device__ void Quaternion::setLookVector(Vec3 other) {
    this->Look = other;
}

__host__ __device__ Quaternion Quaternion::operator+(const Vec3& other) const{
    Quaternion Out = *this;
    Out.Pos += other;
    return Out;
}

__host__ __device__ Quaternion& Quaternion::operator+=(const Vec3& other) {
    this->Pos += other;
    return *this;
}

__host__ __device__ Quaternion& Quaternion::rotate(const Vec3& other, double deg) {
    Vec3 UnitVec = other.unitVector();
    float ux = UnitVec.getX();
    float uy = UnitVec.getY();
    float uz = UnitVec.getZ();

    float angleCos = cos(deg);
    float angleSin = sin(deg);
    
    Vec3 RotateX = Vec3(angleCos + ux*ux*(1-angleCos), ux*uy*(1-angleCos) - uz*angleSin, ux*uz*(1-angleCos)+uy*angleSin);
    Vec3 RotateY = Vec3(uy*ux*(1-angleCos) + uz*angleSin,angleCos + uy*uy*(1-angleCos),uy*uz*(1-angleCos)-ux*angleSin);
    Vec3 RotateZ = Vec3(uz*ux*(1-angleCos)-uy*angleSin,uz*uy*(1-angleCos)+ux*angleSin,angleCos+uz*uz*(1-angleCos));

    this->Up = Vec3(this->Up.dot(RotateX),this->Up.dot(RotateY),this->Up.dot(RotateZ));
    this->Right = Vec3(this->Right.dot(RotateX),this->Right.dot(RotateY),this->Right.dot(RotateZ));
    this->Look = Vec3(this->Look.dot(RotateX),this->Look.dot(RotateY),this->Look.dot(RotateZ));

    return *this;
}

std::ostream& operator<<(std::ostream& os, const Quaternion& CFrame) {
    Vec3 Pos = CFrame.getPos();
    Vec3 Up = CFrame.getUpVector();
    Vec3 Right = CFrame.getRightVector();
    Vec3 Look = CFrame.getLookVector();

    os << "Position: (" << Pos << ", UpVector: " << Up << ", RightVector: " << Right << ", LookVector: " << Look << "\n";
    return os;
}