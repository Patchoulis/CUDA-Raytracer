#pragma once

#include "triangle.h"
#include "quaternion.h"
#include <vector>

class Object {
    public:
        std::vector<Triangle> Triangles;
        Quaternion CFrame;
        Object(std::vector<Triangle> Triangles, Quaternion CFrame = Quaternion(Vec3(0,0,0)) );
        void addTri(Vec3 p1, Vec3 p2, Vec3 p3);
        void setCFrame(Quaternion CFrame);
        void setPos(Vec3 Pos);
};

Object makeCuboid(Vec3 Size, Quaternion CFrame, Material material = Material(Vec3(200,200,200)));