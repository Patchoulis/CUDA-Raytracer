#include "camera.h"
#include <cmath>
#include <cstring>
#include <utility>

Camera::Camera(World* world, Vec3 Pos,int ViewWidth,int ViewHeight,int fromOrigin, float FOV) : CFrame(Quaternion(Pos)), ViewWidth(ViewWidth), ViewHeight(ViewHeight), fromOrigin(fromOrigin), FOV(FOV), View(CameraViewport(this->CFrame, this->ViewWidth, this->ViewHeight, this->FOV, this->fromOrigin)), world(world) {}

Camera::Camera(World* world, Quaternion Coords,int ViewWidth,int ViewHeight,int fromOrigin, float FOV) : CFrame(Coords), ViewWidth(ViewWidth), ViewHeight(ViewHeight), fromOrigin(fromOrigin), FOV(FOV), View(CameraViewport(this->CFrame, this->ViewWidth, this->ViewHeight, this->FOV, this->fromOrigin)), world(world) {}

Quaternion& Camera::move(const Vec3& other) {
    this->CFrame += other;
    this->resetView();
    return this->CFrame;
}

void Camera::setCFrame(Quaternion Pos) {
    this->CFrame = Pos;
    this->resetView();
}

Quaternion& Camera::getCFrame() {
    return this->CFrame;
}

Vec3 Camera::getPos() {
    return this->CFrame.getPos();
}

Quaternion& Camera::rotate(const Vec3& other, double deg) {
    this->CFrame.rotate(other,deg);
    this->resetView();
    return this->CFrame;
}

void Camera::resetView() {
    this->View.SetView(this->CFrame, this->ViewWidth, this->ViewHeight, this->FOV, this->fromOrigin);
}

Vec3 calcRightVector(Quaternion Origin, float FOV, int fromOrigin, int Width, int Height) {
    float imageAspectRatio = Width/Height;
    Vec3 Right = Origin.getRightVector().unitVector();

    return Right * tan(FOV/2) * imageAspectRatio;
}

Vec3 calcUpVector(Quaternion Origin, float FOV, int fromOrigin, int Width, int Height) {
    Vec3 Up = Origin.getUpVector().unitVector();

    return Up * tan(FOV/2);
}

#define COSPRECISION 100
CameraViewport::CameraViewport(Quaternion Origin,int Width, int Height, float FOV, int fromOrigin) : Origin(Origin), Width(Width), Height(Height), unitUp(calcUpVector(Origin,FOV,fromOrigin, Width, Height)), unitRight(calcRightVector(Origin,FOV,fromOrigin, Width, Height)) {

}

void CameraViewport::SetView(Quaternion Origin,int Width, int Height, float FOV, int fromOrigin) {
    this->Origin = Origin;
    Vec3 Right = Origin.getRightVector().unitVector();
    Vec3 Up = Origin.getUpVector().unitVector();

    float imageAspectRatio = Width/Height;

    this->unitRight = Right * imageAspectRatio * tan(FOV/2);
    this->unitUp = Up * tan(FOV/2);
}

__host__ __device__ float NormalDist(const float& alpha, const Vec3& N, const Vec3& H) {
    float num = pow(alpha, 2.0);

    float NDotH = max(N.dot(H),0.0f);
    float den = M_PI * pow(pow(NDotH, 2.0) * (pow(alpha, 2.0) - 1.0) + 1.0, 2.0);
    den = max(den,EPSILON);
    return num/den;
}

__host__ __device__ float G1(const float& alpha, const Vec3& N, const Vec3& X) {
    float num = max(N.dot(X),0.0f);

    float k = alpha/2.0f;
    float den = max(N.dot(X),0.0f) * (1.0f - k) + k;
    den = max(den,EPSILON);
    return num/den;
}

__host__ __device__ Vec3 Fresnel(const Vec3& F0, const Vec3& V, const Vec3& H) {
    return F0 + (Vec3(1.0f,1.0f,1.0f) - F0) * pow(1 - max(V.dot(H), 0.0f), 5);
}

__host__ __device__ float G(const float& alpha, const Vec3& N, const Vec3& V, const Vec3& L) {
    return G1(alpha, N, V) * G1(alpha, N, L);
}

__host__ __device__ Vec3 PBR(const Vec3& Hit, const Vec3& V, const PointLight* lights, int lightCount, Triangle& hitTri, Vec3& Ambience, BVHNode* tree, Triangle* tris, uint* triIndexes) {
    const Material& mat = hitTri.getMaterial();
    const Vec3& N = hitTri.getNorm();
    Vec3 Light = mat.Emissivity;

    Vec3 lambert = (mat.Albedo/255) * M_1_PI;

    for (int i = 0; i < lightCount; i++) {
        Ray RenderRay = Ray(Hit, (lights[i].Pos - Hit).unitVector());
        IntersectBVH(RenderRay, tree, tris, triIndexes);
        if (RenderRay.hasHit()) {
            continue;
        }

        Vec3 L = (lights[i].Pos-Hit).unitVector();
        const Vec3 H = (V + L).unitVector();

        Vec3 Ks = Fresnel(mat.F0, V, H);
        Vec3 Kd = Vec3(1.0f,1.0f,1.0f) - Ks;

        Vec3 cookTorrenceNum = Ks * NormalDist(mat.Roughness, N, H) * G(mat.Roughness, N, V, L);
        float cookTorrenceDen = max(4.0f * max(N.dot(V), 0.0f) * max(N.dot(L), 0.0f),EPSILON);
        Vec3 cookTorrence = cookTorrenceNum / cookTorrenceDen;
        Vec3 BDRF = Kd * lambert + cookTorrence;

        Light = Light + BDRF * lights[i].Color * max(N.dot(L), 0.0f);
    }

    Vec3 Kd = Vec3(1.0f,1.0f,1.0f) - mat.F0;

    Vec3 cookTorrenceNum = mat.F0 * NormalDist(mat.Roughness, N, V);
    float ambientDot = max(N.dot(V), 0.0f);
    float cookTorrenceDen = max(4.0f * ambientDot * ambientDot,EPSILON);
    Vec3 cookTorrence = cookTorrenceNum / cookTorrenceDen;
    Vec3 BDRF = Kd * lambert + cookTorrence;
    Light = Light + BDRF * Ambience * max(N.dot(V), 0.0f);


    
    return Light;
}

__global__ void rayTraceKernel(CameraViewport* View, uint32_t* screenBuffer, BVHNode* tree, Triangle* tris, uint* triIndexes, PointLight* lights, int lightCount, Vec3 CFramePos, int MaxDist, int maxBounces, Vec3 ambience) {
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    int screenWidth = View->Width;
    int screenHeight = View->Height;

    int index = X + screenWidth*Y;

    if (X >= screenWidth || Y >= screenHeight) return;

    Vec3 newPos = View->Origin.getPos() + View->unitRight * (__fdividef(2 * (X + 0.5), screenWidth) - 1) + View->unitUp * (1 - __fdividef(2 * (Y + 0.5), screenHeight)) + View->Origin.getLookVector();
    Ray RenderRay = Ray(View->Origin.getPos(), (newPos - View->Origin.getPos()).unitVector());

    IntersectBVH(RenderRay, tree, tris, triIndexes);
    if (!RenderRay.hasHit()) {
        screenBuffer[index] = 0x000000FF; //SKYBOX
    } else {
        Triangle& hitTri = tris[RenderRay.hit.hit];
        Vec3 Pos = RenderRay.getPos() + RenderRay.getDirection() * RenderRay.hit.t;
        screenBuffer[index] = PBR(Pos,-RenderRay.getDirection(), lights, lightCount, hitTri, ambience, tree, tris, triIndexes).toUint32();
    }
}

// Temporary 
#define MAXDIST 60
#define AMBIENT Vec3(175,175,175)
#define MAXBOUNCES 1
#define THREADSPERBLOCK 32

void Camera::raytrace(Viewport& screen) {
    CameraViewport View = this->View;
    int bufferSize = View.Width * View.Height;
    uint32_t* screenBuffer;
    cudaMalloc(&screenBuffer, bufferSize * sizeof(uint32_t));

    CameraViewport* CamView;
    cudaMalloc(&CamView, sizeof(CameraViewport));
    cudaMemcpy(CamView, &View, sizeof(CameraViewport), cudaMemcpyHostToDevice);

    dim3 Grid(ceil((1.0*View.Width)/32), ceil((1.0*View.Height)/16), 1);
    dim3 Blocks(32, 16, 1);

    rayTraceKernel<<<Grid, Blocks>>>(CamView, screenBuffer, this->world->BVHNodes, this->world->Tris, this->world->TriIndexes, this->world->PointLights, this->world->LightCount, this->CFrame.getPos(), MAXDIST, MAXBOUNCES,AMBIENT);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Synchronize Problem occured: " << err << "\n";
    }

    err = cudaMemcpy(screen.lockAndGetPixels(),screenBuffer,bufferSize * sizeof(uint32_t),cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "MemCpy Problem occured: " << err << "\n";
    }

    screen.setDisplay();
    screen.Update();

    cudaFree(screenBuffer);
    cudaFree(CamView);
}

CameraViewport::~CameraViewport() {
}
