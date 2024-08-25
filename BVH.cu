#include "BVH.h"
#include <cmath>
#include <iostream>
#include <utility>

BVHNode::BVHNode(Vec3 aabbMin, Vec3 aabbMax, uint leftChild, uint firstPrim, uint primCount) :
    aabbMin(aabbMin), aabbMax(aabbMax), leftChild(leftChild), firstPrim(firstPrim), primCount(primCount)
{}

BVHTree::BVHTree(std::vector<Object>& objects) : Tris(this->getTrisFromObjVec(objects)), TriCount(this->calcTriCount(objects)), TriIndexes(new uint[this->TriCount]), Tree(new BVHNode[this->TriCount * 2 - 1]), nodesUsed(1), rootIndex(0) {
    printf("ACTUALLY RAN\n");
    for (uint i = 0; i < this->TriCount; i++) {
        this->TriIndexes[i] = i;
    }
    this->Tree[this->rootIndex].leftChild = 0;
    this->Tree[this->rootIndex].firstPrim = 0;
    this->Tree[this->rootIndex].primCount = this->TriCount;
    this->UpdateNodeBounds(this->rootIndex);
    this->Subdivide(this->rootIndex);
}

uint BVHTree::calcTriCount(std::vector<Object>& objects) const {
    uint Total = 0;
    for (const auto& obj : objects) {
        Total += obj.Triangles.size();
    }
    return Total;
}

Triangle* BVHTree::getTrisFromObjVec(std::vector<Object>& objects) const {
    uint numTris = 0;
    for (const auto& obj : objects) {
        numTris += obj.Triangles.size();
    }
    Triangle* Tris = static_cast<Triangle*>(operator new[](numTris * sizeof(Triangle)));
    uint ind = 0;
    for (uint i = 0; i < objects.size(); i++) {
        for (uint v = 0; v < objects[i].Triangles.size(); v++) {
            new (Tris + ind) Triangle(objects[i].Triangles[v]);
            ind += 1;
        }
    }
    return Tris;
}

Triangle* BVHTree::createDeviceTris() const {
    Triangle* deviceTris = nullptr;
    cudaMalloc((void**) &deviceTris, this->TriCount * sizeof(Triangle));
    cudaMemcpy(deviceTris, this->Tris, this->TriCount * sizeof(Triangle), cudaMemcpyHostToDevice);

    return deviceTris;
}

BVHNode* BVHTree::createDeviceBVHNodes() const {
    BVHNode* deviceBVH = nullptr;
    cudaMalloc((void**) &deviceBVH, (this->TriCount * 2 - 1) * sizeof(BVHNode));
    cudaMemcpy(deviceBVH, this->Tree, (this->TriCount * 2 - 1) * sizeof(BVHNode), cudaMemcpyHostToDevice);

    return deviceBVH;
}

uint* BVHTree::createDeviceTriIndexes() const {
    uint* DeviceTriIndexes = nullptr;
    cudaMalloc((void**) &DeviceTriIndexes, this->TriCount * sizeof(uint));
    cudaMemcpy(DeviceTriIndexes, this->TriIndexes, this->TriCount * sizeof(uint), cudaMemcpyHostToDevice);

    return DeviceTriIndexes;
}

__host__ __device__ bool BVHNode::isLeaf() {
    return this->primCount > 0;
}

void BVHTree::UpdateNodeBounds(uint nodeIdx) {
    BVHNode& node = this->Tree[nodeIdx];
    node.aabbMin =  Vec3(1e30f,1e30f,1e30f);
    node.aabbMax =  Vec3(-1e30f,-1e30f,-1e30f);

    for (uint first = node.firstPrim, i = 0; i < node.primCount; i++)
    {
        uint Ind = this->TriIndexes[first + i];
        Triangle& leafTri = this->Tris[Ind];
        node.aabbMin.x = min( node.aabbMin.x, leafTri.getMinX() );
        node.aabbMin.y = min( node.aabbMin.y, leafTri.getMinY() );
        node.aabbMin.z = min( node.aabbMin.z, leafTri.getMinZ() );

        node.aabbMax.x = max( node.aabbMax.x, leafTri.getMaxX() );
        node.aabbMax.y = max( node.aabbMax.y, leafTri.getMaxY() );
        node.aabbMax.z = max( node.aabbMax.z, leafTri.getMaxZ() );
    }
}

void swap(uint& a, uint& b) {
    uint aCopy = a;
    a = b;
    b = aCopy;
}

void BVHTree::Subdivide(uint nodeIdx) {
    // terminate recursion
    BVHNode& node = this->Tree[nodeIdx];
    if (node.primCount <= 2)  {
        return;
    }

    float extentX = node.aabbMax.x - node.aabbMin.x;
    float extentY = node.aabbMax.y - node.aabbMin.y;
    float extentZ = node.aabbMax.z - node.aabbMin.z;

    Vec3 extent = Vec3(extentX, extentY, extentZ);

    uint axis = 0;
    if (extent.y > extent.x) {
        axis = 1;
    }
    if ( extent.z > extent[axis] ) {
        axis = 2;
    }
    float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
    // In-place partition
    int i = node.firstPrim;
    int j = i + node.primCount - 1;
    while (i <= j)
    {
        if (Tris[TriIndexes[i]].getCentroid()[axis] < splitPos) {
            i++;
        }
        else {
            swap( TriIndexes[i], TriIndexes[j--] );
        }
    }
    // Abort split if one of the sides is empty
    int leftCount = i - node.firstPrim;
    if (leftCount == 0 || leftCount == node.primCount) {
        return;
    }
    // Create child nodes
    int leftChildIdx = this->nodesUsed++;
    int rightChildIdx = this->nodesUsed++;
    this->Tree[leftChildIdx].firstPrim = node.firstPrim;
    this->Tree[leftChildIdx].primCount = leftCount;
    this->Tree[rightChildIdx].firstPrim = i;
    this->Tree[rightChildIdx].primCount = node.primCount - leftCount;
    node.leftChild = leftChildIdx;
    node.primCount = 0;
    UpdateNodeBounds( leftChildIdx );
    UpdateNodeBounds( rightChildIdx );

    Subdivide( leftChildIdx );
    Subdivide( rightChildIdx );
}



__host__ __device__ void IntersectBVH(Ray& ray, BVHNode*& Tree, Triangle*& Tris, uint*& TriIndexes) {
    uint stack[64];
    uint stackPtr = 0;

    stack[stackPtr++] = 0;
    while (stackPtr > 0) {
        uint nodeIdx = stack[--stackPtr];
        BVHNode& node = Tree[nodeIdx];
        if (!IntersectAABB( ray, node.aabbMin, node.aabbMax )){
            continue;
        }
        if (node.isLeaf())
        {   
            for (uint i = 0; i < node.primCount; i++ ) {
                IntersectTri(ray, Tris[TriIndexes[node.firstPrim + i]], TriIndexes[node.firstPrim + i]);
            }
        } else {
            stack[stackPtr++] = node.leftChild+1;
            stack[stackPtr++] = node.leftChild;
        }
    }
}

__host__ __device__ bool IntersectAABB(const Ray& ray, const Vec3& bmin, const Vec3& bmax) {
    float tx1 = (bmin.x - ray.getPos().x)/ray.getDirection().x;
    float tx2 = (bmax.x - ray.getPos().x)/ray.getDirection().x;
    //printf("TEST: %f, %f\n",tx1,ray.getPos().x);

    float tmin = min( tx1, tx2 );
    float tmax = max( tx1, tx2 );

    float ty1 = (bmin.y - ray.getPos().y)/ray.getDirection().y;
    float ty2 = (bmax.y - ray.getPos().y)/ray.getDirection().y;

    tmin = max( tmin, min( ty1, ty2 ));
    tmax = min( tmax, max( ty1, ty2 ));

    float tz1 = (bmin.z - ray.getPos().z)/ray.getDirection().z;
    float tz2 = (bmax.z - ray.getPos().z)/ray.getDirection().z;

    tmin = max( tmin, min( tz1, tz2 ));
    tmax = min( tmax, max( tz1, tz2 ));

    //printf("DATA: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",tmax, tmin, ray.hit.t, tx1, tx2, ty1, ty2, tz1, tz2);

    return tmax >= tmin && tmin < ray.hit.t && tmax > 0;
}


__host__ __device__ void IntersectTri(Ray& ray, const Triangle& tri, const uint instPrim ) {
	const Vec3 edge1 = tri.p2 - tri.p1;
	const Vec3 edge2 = tri.p3 - tri.p1;
	const Vec3 h = ray.getDirection().cross(edge2);
	const float a = edge1.dot(h);
	if (a < EPSILON) {
        return;
    }
	const float f = 1 / a;
	const Vec3 s = ray.getPos() - tri.p1;
	const float u = s.dot(h) * f;
	if (u < 0 || u > 1) {
        return;
    }
	const Vec3 q = s.cross(edge1);
	const float v = ray.getDirection().dot(q) * f;
	if (v < 0 || u + v > 1) {
        return;
    }
	const float t = edge2.dot(q) * f;
	if (t > EPSILON && t < ray.hit.t) {
        ray.hit.t = t, ray.hit.u = u,
		ray.hit.v = v, ray.hit.hit = instPrim;
    }
}

BVHTree::~BVHTree() {
    std::cout << "FREED TREE\n";
    for (uint i = 0; i < this->TriCount; i++) {
        Tris[i].~Triangle();
    }
    for (uint i = 0; i < this->TriCount * 2 - 1; i++) {
        Tree[i].~BVHNode();
    }
    operator delete[](Tris);
    operator delete[](Tree);
    operator delete[](TriIndexes);
}