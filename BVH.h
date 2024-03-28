#include "Vec3.h"
#include "object.h"
#include "ray.h"
#include "material.h"
#include "triangle.h"
#include <utility>
#include <vector>

#pragma once

#define EPSILON 0.00001

struct BVHNode {
    Vec3 aabbMin, aabbMax;
    uint leftChild;
    uint firstPrim, primCount;
    __host__ __device__ bool isLeaf();
    BVHNode(Vec3 aabbMin = Vec3(0.0f,0.0f,0.0f), Vec3 aabbMax = Vec3(0.0f,0.0f,0.0f), uint leftChild = 0, uint firstPrim = 0, uint primCount = 0);
};

class BVHTree {
    private:
        Triangle* Tris;
        uint TriCount;
        uint* TriIndexes;
        BVHNode* Tree;
        uint nodesUsed, rootIndex;
        Triangle* getTrisFromObjVec(std::vector<Object>& objects) const;
        uint calcTriCount(std::vector<Object>& objects) const;
    public:
        BVHTree(std::vector<Object>& objects);
        void UpdateNodeBounds(std::vector<Triangle>& Tris);
        void UpdateNodeBounds(uint nodeIdx);
        void Subdivide(uint nodeIdx);
        Triangle* createDeviceTris() const;
        uint* createDeviceTriIndexes() const;
        BVHNode* createDeviceBVHNodes() const;
        uint getNodesUsed() const;
        ~BVHTree();
};

__host__ __device__ void IntersectBVH(Ray& ray, BVHNode*& Tree, Triangle*& Tris, uint*& TriIndexes, uint nodeIdx = 0);
__host__ __device__ bool IntersectAABB(const Ray& ray, const Vec3& bmin, const Vec3& bmax);
__host__ __device__ void IntersectTri(Ray& ray, const Triangle& tri, const uint instPrim );