#include "object.h"
#include "pointlight.h"
#include <vector>

#pragma once

class World {
    private:
        std::vector<Triangle> Tris;
        std::vector<Object> Renderable;
        std::vector<PointLight> PointLights;
        void addTris(Object& newObj);
        std::vector<Triangle> getTrisFromObjVec(std::vector<Object>& objects) const;
    public:
        World(std::vector<Object> Renderable);
        void AddObj(Object newObj);
        void AddPointLight(PointLight light);
        std::pair<const Triangle*,int> getTris(Quaternion& Cam) const;
        std::pair<const PointLight*,int> getPointLights(Quaternion& Cam) const;
};