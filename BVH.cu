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
        node.aabbMin.setX(min( node.aabbMin.getX(), leafTri.getMinX() ));
        node.aabbMin.setY(min( node.aabbMin.getY(), leafTri.getMinY() ));
        node.aabbMin.setZ(min( node.aabbMin.getZ(), leafTri.getMinZ() ));

        node.aabbMax.setX(max( node.aabbMax.getX(), leafTri.getMaxX() ));
        node.aabbMax.setY(max( node.aabbMax.getY(), leafTri.getMaxY() ));
        node.aabbMax.setZ(max( node.aabbMax.getZ(), leafTri.getMaxZ() ));

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

    float extentX = node.aabbMax.getX() - node.aabbMin.getX();
    float extentY = node.aabbMax.getY() - node.aabbMin.getY();
    float extentZ = node.aabbMax.getZ() - node.aabbMin.getZ();

    Vec3 extent = Vec3(extentX, extentY, extentZ);

    uint axis = 0;
    if (extent.getY() > extent.getX()) {
        axis = 1;
    }
    if ( extent.getZ() > extent[axis] ) {
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



__host__ __device__ void IntersectBVH(Ray& ray, BVHNode*& Tree, Triangle*& Tris, uint*& TriIndexes, const uint nodeIdx) {
    BVHNode& node = Tree[nodeIdx];
    //printf("AAB: %f, %f, %f, %d\n",node.aabbMin.getX(),node.aabbMin.getY(),node.aabbMin.getZ(), node.primCount);
    //printf("RAY: %f, %f, %f\n",ray.getPos().getX(),ray.getPos().getY(),ray.getPos().getZ());
    if (!IntersectAABB( ray, node.aabbMin, node.aabbMax )){
        //printf("AAB: %f, %f, %f, %d\n",node.aabbMin.getX(),node.aabbMin.getY(),node.aabbMin.getZ(), node.primCount);
        //printf("RAY: %f, %f, %f\n",ray.getPos().getX(),ray.getPos().getY(),ray.getPos().getZ());
        return;
    }
    if (node.isLeaf())
    {   
        for (uint i = 0; i < node.primCount; i++ ) {
            IntersectTri(ray, Tris[TriIndexes[node.firstPrim + i]], TriIndexes[node.firstPrim + i]);
        }
    }
    else
    {
        IntersectBVH(ray, Tree, Tris, TriIndexes, node.leftChild);
        IntersectBVH(ray, Tree, Tris, TriIndexes, node.leftChild + 1);
    }
}

__host__ __device__ bool IntersectAABB(const Ray& ray, const Vec3& bmin, const Vec3& bmax) {
    float tx1 = (bmin.getX() - ray.getPos().getX())/ray.getDirection().getX();
    float tx2 = (bmax.getX() - ray.getPos().getX())/ray.getDirection().getX();
    //printf("TEST: %f, %f\n",tx1,ray.getPos().getX());

    float tmin = min( tx1, tx2 );
    float tmax = max( tx1, tx2 );

    float ty1 = (bmin.getY() - ray.getPos().getY())/ray.getDirection().getY();
    float ty2 = (bmax.getY() - ray.getPos().getY())/ray.getDirection().getY();

    tmin = max( tmin, min( ty1, ty2 ));
    tmax = min( tmax, max( ty1, ty2 ));

    float tz1 = (bmin.getZ() - ray.getPos().getZ())/ray.getDirection().getZ();
    float tz2 = (bmax.getZ() - ray.getPos().getZ())/ray.getDirection().getZ();

    tmin = max( tmin, min( tz1, tz2 ));
    tmax = min( tmax, max( tz1, tz2 ));

    //printf("DATA: %f, %f, %f, %f, %f, %f, %f, %f, %f\n",tmax, tmin, ray.hit.t, tx1, tx2, ty1, ty2, tz1, tz2);

    return tmax >= tmin && tmin < ray.hit.t && tmax > 0;
}


__host__ __device__ void IntersectTri(Ray& ray, const Triangle& tri, const uint instPrim ) {
	const Vec3 edge1 = tri.getGlobalV2() - tri.getGlobalV1();
	const Vec3 edge2 = tri.getGlobalV3() - tri.getGlobalV1();
	const Vec3 h = ray.getDirection().cross(edge2);
	const float a = edge1.dot(h);
	if (a < EPSILON) {
        return;
    }
	const float f = 1 / a;
	const Vec3 s = ray.getPos() - tri.getGlobalV1();
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