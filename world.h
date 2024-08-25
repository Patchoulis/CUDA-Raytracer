#include "object.h"
#include "pointlight.h"
#include "BVH.h"
#include "skybox.h"
#include <vector>

#pragma once

class World {
    public:
        BVHTree Tree;
        Triangle* Tris;
        uint* TriIndexes;
        BVHNode* BVHNodes;
        PointLight* PointLights;
        Skybox sky;
        uint LightCount;
        World(std::vector<Object> Renderable, std::vector<PointLight> PointLights,Skybox& skybox);
        ~World();
};