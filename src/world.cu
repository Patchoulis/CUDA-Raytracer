#include "world.h"
#include <utility>

PointLight* getDevicePointLights(std::vector<PointLight> PointLights) {
    PointLight* devicePointLights = nullptr;
    cudaMalloc((void**) &devicePointLights, PointLights.size()* sizeof(PointLight));
    cudaMemcpy(devicePointLights, PointLights.data(), PointLights.size() * sizeof(PointLight), cudaMemcpyHostToDevice);

    return devicePointLights;
}

World::World(std::vector<Object> Renderable, std::vector<PointLight> PointLights) : Tree(BVHTree(Renderable)), Tris(this->Tree.createDeviceTris()), TriIndexes(this->Tree.createDeviceTriIndexes()),
    BVHNodes(this->Tree.createDeviceBVHNodes()), PointLights(getDevicePointLights(PointLights)), LightCount(PointLights.size()) {}

World::~World() {
    std::cout << "FREED WORLD\n";
    cudaFree(this->BVHNodes);
    cudaFree(this->Tris);
    cudaFree(this->TriIndexes);
}