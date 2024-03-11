#include "object.h"

Object::Object(std::vector<Triangle> Triangles, Quaternion CFrame) : Triangles(Triangles), CFrame(CFrame) {}

void Object::addTri(Vec3 p1, Vec3 p2, Vec3 p3) {
    this->Triangles.push_back(Triangle(p1,p2,p3,this->CFrame));
}

void Object::setCFrame(Quaternion CFrame) {
    this->CFrame = CFrame;
}

void Object::setPos(Vec3 Pos) {
    Vec3 Diff = this->CFrame.getPos() - Pos;
    this->CFrame.setPos(Pos);
    for (Triangle& Tri: this->Triangles) {
        Tri.updatePos(Diff);
    }
}

Object makeCuboid(Vec3 Size, Quaternion CFrame, Material material) {

    Vec3 bfl = Vec3(-Size.getX()/2,-Size.getY()/2,-Size.getZ()/2);
    Vec3 bfr = Vec3(Size.getX()/2,-Size.getY()/2,-Size.getZ()/2);
    Vec3 tfr = Vec3(Size.getX()/2,Size.getY()/2,-Size.getZ()/2);
    Vec3 tfl = Vec3(-Size.getX()/2,Size.getY()/2,-Size.getZ()/2);

    Vec3 bbl = Vec3(-Size.getX()/2,-Size.getY()/2,Size.getZ()/2);
    Vec3 bbr = Vec3(Size.getX()/2,-Size.getY()/2,Size.getZ()/2);
    Vec3 tbr = Vec3(Size.getX()/2,Size.getY()/2,Size.getZ()/2);
    Vec3 tbl = Vec3(-Size.getX()/2,Size.getY()/2,Size.getZ()/2);

    std::vector<Triangle> SquareTris{ Triangle(bfr,bfl,tfl,CFrame,material), Triangle(tfr,bfr,tfl,CFrame,material), Triangle(bbl,bbr,tbl,CFrame,material), Triangle(tbl,bbr,tbr,CFrame,material), Triangle(tfl,bfl,bbl,CFrame,material), Triangle(tbl,tfl,bbl,CFrame,material), 
        Triangle(bfr,tfr,bbr,CFrame,material), Triangle(tfr,tbr,bbr,CFrame,material), Triangle(tfr,tfl,tbl,CFrame,material), Triangle(tbr,tfr,tbl,CFrame,material), Triangle(bfl,bfr,bbl,CFrame,material), Triangle(bbl,bfr,bbr,CFrame,material) };
    return Object(SquareTris,CFrame);
}