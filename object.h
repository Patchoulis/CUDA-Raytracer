#include "triangle.h"
#include "quaternion.h"
#include <vector>
#include <fstream>

#pragma once

class Object {
    public:
        std::vector<Triangle> Triangles;
        Quaternion CFrame;
        Object(std::vector<Triangle> Triangles, Quaternion CFrame = Quaternion(Vec3(0,0,0)) );
        void setCFrame(Quaternion CFrame);
        void setPos(Vec3 Pos);
};

Object makeCuboid(Vec3 Size, Quaternion CFrame, Material material = Material(Vec3(200,200,200)));
Object makeMesh(const char*& filename,Quaternion CFrame,Vec3 Size,Material material);