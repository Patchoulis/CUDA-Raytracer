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

CameraViewport::CameraViewport(Quaternion Origin,int Width, int Height, float FOV, int fromOrigin) : Origin(Origin), Width(Width), Height(Height), unitUp(calcUpVector(Origin,FOV,fromOrigin, Width, Height)), unitRight(calcRightVector(Origin,FOV,fromOrigin, Width, Height)) {}

void CameraViewport::SetView(Quaternion Origin,int Width, int Height, float FOV, int fromOrigin) {
    this->Origin = Origin;
    Vec3 Right = Origin.getRightVector().unitVector();
    Vec3 Up = Origin.getUpVector().unitVector();

    float imageAspectRatio = Width/Height;

    this->unitRight = Right * imageAspectRatio * tan(FOV/2);
    this->unitUp = Up * tan(FOV/2);
}

__global__ void rayTraceKernel(CameraViewport* View, uint32_t* screenBuffer, Triangle* renderable, int RenderableCount, PointLight* lights, int lightCount, Vec3 CFramePos, int MaxDist, int maxBounces, float ambience,float shadow) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int screenWidth = View->Width;
    int screenHeight = View->Height;
    if (index >= screenWidth * screenHeight) return;

    int X = index % screenWidth;
    int Y = index / screenWidth;


    Vec3 newPos = View->Origin.getPos() + View->unitRight * (2*((X+0.5)/screenWidth) - 1) + View->unitUp * (1 - 2*((Y+0.5)/screenHeight)) + View->Origin.getLookVector();
    Ray RenderRay = Ray(View->Origin.getPos() , (newPos - View->Origin.getPos() ).unitVector());
    float Intensity = ambience;

    Triangle* CurrTri = nullptr;
    Triangle* LastTri = nullptr;

    screenBuffer[index] = 0x000000FF;

    for (int v = 0; v < maxBounces; v++) {
        float maxed = MaxDist;
        CurrTri = nullptr;

        for (int i = 0; i < RenderableCount; i++) {
            if (LastTri == &renderable[i]) {
                continue;
            }
            RayIntersectResult Res = renderable[i].rayIntersect(RenderRay);
            if (Res.hit && Res.t < maxed && Res.t > EPSILON) {
                maxed = Res.t;
                CurrTri = &renderable[i];
            }
        }

        LastTri = CurrTri;
        
        if (CurrTri != nullptr) {
            RenderRay = Ray(RenderRay.getPosAtDist(maxed), (RenderRay.getDirection() - CurrTri->getNorm()*2.0f*CurrTri->getNorm().dot(RenderRay.getDirection())).unitVector());
        } else {
            Ray OldRay = RenderRay;

            for (int i = 0; i < lightCount; i++) {
                PointLight light = lights[i];
                RenderRay = Ray(RenderRay.getPos(), (light.Pos-RenderRay.getPos()));

                // Shadow Calculations
                bool NotHit = true;
                for (int i = 0; i < RenderableCount; i++) {
                    if (LastTri == &renderable[i]) {
                        continue;
                    }
                    RayIntersectResult Res = renderable[i].rayIntersect(RenderRay);
                    if (Res.hit && Res.t > EPSILON) {
                        NotHit = false;
                        break;
                    }
                }
                if (NotHit) {
                    Intensity = min(max((light.Pos-OldRay.getPos()).unitVector().dot(OldRay.Dir), 0.0f) + Intensity, 1.0f);
                } else {
                    Intensity = shadow;
                }


            }
            screenBuffer[index] = (CurrTri != nullptr) ? CurrTri->getColor().toUint32( Intensity ) : 0x000000FF;
        }
    }
    
    if (CurrTri != nullptr) {

        Ray OldRay = RenderRay;
        bool NotHit = true;

        for (int i = 0; i < lightCount; i++) {
            PointLight light = lights[i];
            RenderRay = Ray(RenderRay.getPos(), (light.Pos-RenderRay.getPos()));

            // Shadow Calculations
            NotHit = true;
            for (int i = 0; i < RenderableCount; i++) {
                if (LastTri == &renderable[i]) {
                    continue;
                }
                RayIntersectResult Res = renderable[i].rayIntersect(RenderRay);
                if (Res.hit && Res.t > EPSILON) {
                    NotHit = false;
                    break;
                }
            }
            if (NotHit) { //CHANGE LATER
                Intensity = min(max((light.Pos-OldRay.getPos()).unitVector().dot(OldRay.Dir), 0.0f) + Intensity, 1.0f);
            } else {
                Intensity = shadow;
            }


        }
        screenBuffer[index] = (CurrTri != nullptr) ? CurrTri->getColor().toUint32( Intensity ) : 0x000000FF;
    } 
    
}

// Temporary 
#define MAXDIST 200
#define AMBIENT 0.25
#define SHADOW 0.1
#define MAXBOUNCES 1
#define THREADSPERBLOCK 512

void Camera::raytrace(Viewport& screen) {
    CameraViewport View = this->View;
    int bufferSize = View.Width * View.Height;
    uint32_t* screenBuffer;
    cudaMalloc(&screenBuffer, bufferSize * sizeof(uint32_t));

    CameraViewport* CamView;
    cudaMalloc(&CamView, sizeof(CameraViewport));
    cudaMemcpy(CamView, &View, sizeof(CameraViewport), cudaMemcpyHostToDevice);

    Triangle* Tris;
    std::pair<const Triangle*,int> TriList = this->world->getTris(View.Origin);
    cudaMalloc(&Tris, TriList.second * sizeof(Triangle));
    cudaMemcpy(Tris, TriList.first, TriList.second * sizeof(Triangle), cudaMemcpyHostToDevice);

    PointLight* Lights;
    std::pair<const PointLight*,int> LightList = this->world->getPointLights(View.Origin);
    cudaMalloc(&Lights, LightList.second * sizeof(PointLight));
    cudaMemcpy(Lights, LightList.first, LightList.second * sizeof(PointLight), cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADSPERBLOCK; 
    int blocks = (bufferSize + threadsPerBlock - 1) / threadsPerBlock;

    rayTraceKernel<<<blocks, threadsPerBlock>>>(CamView, screenBuffer, Tris, TriList.second, Lights, LightList.second, this->CFrame.getPos(), MAXDIST, MAXBOUNCES,AMBIENT, SHADOW);

    cudaDeviceSynchronize();

    cudaError_t err = cudaMemcpy(screen.lockAndGetPixels(),screenBuffer,bufferSize * sizeof(uint32_t),cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "Problem occured: " << err << "\n";
    }

    screen.setDisplay();
    screen.Update();

    cudaFree(screenBuffer);
    cudaFree(CamView);
    cudaFree(Tris);
}

CameraViewport::~CameraViewport() {
}
