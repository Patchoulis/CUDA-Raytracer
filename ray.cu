#include "ray.h"
#include "Vec3.h"

Ray::Ray(Vec3 Pos,Vec3 Dir) : Pos(Pos), Dir(Dir) {}

Ray Ray::operator*(const float other) const {
    return Ray(this->getPos(),this->getDirection()*other);
}

Ray& Ray::operator*=(const float other) {
    this->Dir*=other;
    return *this;
}

Vec3 Ray::getPos() const {
    return this->Pos;
}

Vec3 Ray::getDirection() const {
    return this->Dir;
}

Vec3 Ray::getPosAtDist(float dist) const {
    return this->Dir * dist + this->Pos;
}