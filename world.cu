#include "world.h"
#include <utility>

World::World(std::vector<Object> Renderable) : Renderable(Renderable), Tris(getTrisFromObjVec(Renderable)) {}

std::vector<Triangle> World::getTrisFromObjVec(std::vector<Object>& objects) const {
    std::vector<Triangle> AllTris;
    for (const auto& obj : objects) {
        AllTris.insert(AllTris.end(), obj.Triangles.begin(), obj.Triangles.end());
    }
    return AllTris;
}

void World::AddObj(Object newObj) {
    this->Renderable.push_back(newObj);
    this->addTris(newObj);
}

void World::AddPointLight(PointLight light) {
    this->PointLights.push_back(light);
}

std::pair<const Triangle*,int> World::getTris(Quaternion& Cam) const { // UPDATE SO IT USES OCTREES FOR TRIANGLES OBTAINED
    return std::make_pair(this->Tris.data(),this->Tris.size());
}

std::pair<const PointLight*,int> World::getPointLights(Quaternion& Cam) const { // UPDATE SO IT USES OCTREES FOR POINTLIGHTS OBTAINED
    return std::make_pair(this->PointLights.data(),this->PointLights.size());
}

void World::addTris(Object& newObj) {
    for (Triangle tri : newObj.Triangles) {
        this->Tris.push_back(tri);
    }
}