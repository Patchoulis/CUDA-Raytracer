#pragma once

#include "object.h"
#include "pointlight.h"
#include "BVH.h"
#include <vector>

class World {
    public:
        BVHTree Tree;
        Triangle* Tris;
        uint* TriIndexes;
        BVHNode* BVHNodes;
        PointLight* PointLights;
        uint LightCount;
        World(std::vector<Object> Renderable, std::vector<PointLight> PointLights);
        ~World();
};