#include "camera.h"
#include <cmath>
#include <cstring>
#include <utility>

// Temporary 
#define MAXDIST 60
#define AMBIENT Vec3(0.4,0.4,0.4)
#define MAXBOUNCES 3
#define SAMPLES 60
#define THREADSPERBLOCK 32

__global__ void setup_rand_state(curandState *state, uint Width, uint Height, unsigned long seed) {
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (X + Width * Y);

    if (X >= Width || Y >= Height) return;
    curand_init(seed, index, 0, &state[index]);
}

void Camera::setup(uint Width, uint Height) {
    std::cout << "CREATING A CAMERA \n";
    this->d_randStates = nullptr;

    cudaError_t cudaStatus = cudaMalloc((void**) &(this->d_randStates), Width * Height * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for randStates: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }
    
    dim3 Grid(ceil((1.0 * Width) / 32), ceil((1.0 * Height) / 16), 1);
    dim3 Blocks(32, 16, 1);

    setup_rand_state<<<Grid, Blocks>>>(this->d_randStates, Width, Height, time(NULL));
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Setup for randstate kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(this->d_randStates); // Free memory if the kernel launch fails
        return;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Device Synchronize failed after setting up rand state: %s\n", cudaGetErrorString(cudaStatus));
    }
}


Camera::Camera(World* world, Vec3 Pos,int ViewWidth,int ViewHeight,int fromOrigin, float FOV) : CFrame(Quaternion(Pos)), ViewWidth(ViewWidth), ViewHeight(ViewHeight), fromOrigin(fromOrigin), FOV(FOV), View(CameraViewport(this->CFrame, this->ViewWidth, this->ViewHeight, this->FOV, this->fromOrigin)), world(world) {
    #if SAMPLES != 0
    this->setup(ViewWidth,ViewHeight);
    #endif
}

Camera::Camera(World* world, Quaternion Coords,int ViewWidth,int ViewHeight,int fromOrigin, float FOV) : CFrame(Coords), ViewWidth(ViewWidth), ViewHeight(ViewHeight), fromOrigin(fromOrigin), FOV(FOV), View(CameraViewport(this->CFrame, this->ViewWidth, this->ViewHeight, this->FOV, this->fromOrigin)), world(world) {
    #if SAMPLES != 0
    this->setup(ViewWidth,ViewHeight);
    #endif
}

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

__device__ Vec3 SpecSample(const Vec3& V, curandState& randState, const Triangle& tri) {
    const Material& mat = tri.getMaterial();
    float roughness = mat.Roughness;

    const Vec3& N = tri.getNorm();
    Vec3 XAxis = (tri.getGlobalV1()-tri.getGlobalV2()).unitVector();
    Vec3 ZAxis = N.cross(XAxis);
    Vec3 wo = -Vec3(XAxis.dot(V)*roughness,N.dot(V),ZAxis.dot(V)*roughness).unitVector();

    Vec3 T1 = (wo.getY() < 0.9999) ? wo.cross(Vec3(0,1,0)) : Vec3(1,0,0);
    Vec3 T2 = T1.cross(wo);

    float a = 1.0f / (1.0f + wo.getY());

    float rand1 = curand_uniform(&randState);
    float rand2 = curand_uniform(&randState);
    float r = sqrt(rand1);
    
    float phi = (rand2 < a) ? rand2/a * M_PI : M_PI + (rand2-a)/(1.0f-a) * M_PI;
    float P1 = r * cos(phi);
    float P2 = r * sin(phi) * ((rand2<a) ? 1.0 : wo.getY());

    Vec3 n = T1*P1+T2*P2+wo*sqrt(max(0.0f,1.0f-P1*P1-P2*P2));

    return (XAxis * roughness * n.getX() + ZAxis * roughness * n.getZ() + N * max(0.0f,n.getY())).unitVector();
}

__device__ Vec3 DiffuseSample(curandState& randState, const Triangle& tri) {
    const Vec3& N = tri.getNorm();
    Vec3 XAxis = (tri.getGlobalV1()-tri.getGlobalV2()).unitVector();
    Vec3 ZAxis = N.cross(XAxis);

    float rand1 = curand_uniform(&randState);
    float rand2 = curand_uniform(&randState);
    float r = sqrt(rand1);
    
    float theta = 2 * M_PI * rand2;
    float x = r*cos(theta);
    float y = r*sin(theta);

    Vec3 n = Vec3(x,max(0.0f,sqrt(1-rand1)),y).unitVector();

    return (XAxis * n.getX() + ZAxis * n.getZ() + N * n.getY());
}


__device__ float NormalDist(const float& alpha, const Vec3& N, const Vec3& H) {
    float num = alpha * alpha;

    float NDotH = max(N.dot(H),0.0f);
    float den = M_PI * pow(pow(NDotH, 2.0) * (pow(alpha, 2.0) - 1.0) + 1.0, 2.0);
    den = max(den,EPSILON);
    return num/den;
}

__device__ float GGXMask(const float alpha, const Vec3& N, const Vec3& wo) {
    float NV = max(N.dot(wo),0.0f);

    float k = (alpha * alpha);
    float den = max(sqrt(k + (1.0f-k) * NV * NV) + NV,EPSILON);
    return (2.0f * NV)/den;
}

__device__ float GGXShadow(const float alpha, const Vec3& N, const Vec3& wi, const Vec3& wo) {
    float NL = max(N.dot(wi),0.0f);
    float NV = max(N.dot(wo),0.0f);

    float k = (alpha * alpha);
    float denA = max(sqrt(k + (1.0f-k) * NV * NV),EPSILON);
    float denB = max(sqrt(k + (1.0f-k) * NL * NL),EPSILON);
    return (2.0f * NV * NL)/(denA + denB);
}

__device__ Vec3 Fresnel(const Vec3& F0, const Vec3& V, const Vec3& H) {
    return F0 + (Vec3(1.0f,1.0f,1.0f) - F0) * pow(1 - max(V.dot(H), 0.0f), 5);
}

/*
__device__ float G(const float alpha, const Vec3& N, const Vec3& V, const Vec3& L) {
    return G1(alpha, N, V) * G1(alpha, N, L);
} */

__device__ float FrDiaelectric(const Vec3& N, const Vec3& V, float OriginalIndex, float NewIndex) {
    float cosI = N.dot(V);
    float etaI = OriginalIndex;
    float etaT = NewIndex;

    bool exiting = cosI <= 0;
    if (exiting) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }
    float sinI = sqrt(max(0.0f, 1.0f - cosI * cosI));
    float sinT = etaI / etaT * sinI;

    if (sinT >= 1) {
        return 1.0f;
    } else {
        float cosT = sqrt(max(0.0f, 1.0f - sinT * sinT));
        float Rs = ((etaT * cosI) - (etaI * cosT)) / ((etaT * cosI) + (etaI * cosT));
        float Rp = ((etaI * cosI) - (etaT * cosT)) / ((etaI * cosI) + (etaT * cosT));
        return (Rs * Rs + Rp * Rp) / 2.0f;
    }
}


/*


int idx = id * SAMPLES + blockIdx.y;

    Vec3 a = Vec3(1,1,1);
    Vec3 V,Pos,NewDiffuseSample,H,Ks,Kd,cookTorrenceNum,cookTorrence,Result;
    float cookTorrenceDen;
    
    //#pragma unroll MAXBOUNCES
    for (uint i = 0; i < MAXBOUNCES; i++) {
        V = RenderRay.getDirection();
        Triangle& hitTri = tris[RenderRay.hit.hit];
        Pos = RenderRay.getPos() + RenderRay.getDirection() * RenderRay.hit.t;

        const Material& mat = hitTri.getMaterial();
        const Vec3& N = hitTri.getNorm();
        
        NewDiffuseSample = SpecSample(N,state[idx],hitTri);
        
        H = (V + NewDiffuseSample).unitVector();
        Ks = Fresnel(mat.F0, V, H);
        Kd = Vec3(1.0f,1.0f,1.0f) - Ks;

        cookTorrenceNum = Ks * NormalDist(mat.Roughness, N, H) * G(mat.Roughness, N, V, NewDiffuseSample);
        cookTorrenceDen = max(4.0f * max(N.dot(V), 0.0f) * max(N.dot(NewDiffuseSample), 0.0f),EPSILON);
        cookTorrence = cookTorrenceNum / cookTorrenceDen;

        //printf("HAS HIT: %d %d %d \n",RenderRay.hasHit(), RenderRay.hit.hit, RenderRay.hit.t);

        RenderRay = Ray(Pos, -RenderRay.getDirection() + N * N.dot(RenderRay.getDirection()) * 2);
        Result = mat.Emissivity * a;
        printf("COLOR: %f %f %f",mat.Emissivity.getX(),mat.Emissivity.getY(),mat.Emissivity.getZ());
        a = (mat.Diffuse * Kd + cookTorrence);
        IntersectBVH(RenderRay, tree, tris, triIndexes);
        if (!RenderRay.hasHit()) {
            atomicAddVec3(Light[id], (Result + a * getSkyColor(RenderRay.Dir,skyWidth,skyHeight,skyColors))/SAMPLES);
            return;
        } else {
            atomicAddVec3(Light[id], Result/SAMPLES);
        }
    }



__device__ Vec3 PBR(Ray& RenderRay, BVHNode* tree, Triangle* tris, uint* triIndexes, curandState*& state, uint skyWidth, uint skyHeight, Vec3* skyColors, uint id) {
    IntersectBVH(RenderRay, tree, tris, triIndexes);
    if (RenderRay.hasHit()) {
        #if SAMPLES == 0
            Triangle& hitTri = tris[RenderRay.hit.hit];
            const Material& mat = hitTri.getMaterial();
            return mat.Emissivity;
        #else
            Vec3 Light = Vec3(0,0,0);
            computeDiffuseKernel<<<1, SAMPLES>>>(RenderRay, tris, state, Light, id, tree, triIndexes, skyWidth, skyHeight, skyColors);
            
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("Synchronize Problem occured: %d\n",err);
            }

            return Light;
        #endif
    } else {
        return getSkyColor(RenderRay.Dir,skyWidth,skyHeight,skyColors);
    }
    
    if (RenderRay.hasHit()) {
        if (bounces == 0) {
            Triangle& hitTri = tris[RenderRay.hit.hit];
            const Material& mat = hitTri.getMaterial();
            return mat.Emissivity;
        } else {
            Vec3 V = RenderRay.getDirection();
            Triangle& hitTri = tris[RenderRay.hit.hit];
            Vec3 Pos = RenderRay.getPos() + RenderRay.getDirection() * RenderRay.hit.t;

            const Material& mat = hitTri.getMaterial();
            const Vec3& N = hitTri.getNorm();

            Vec3 Light = mat.Emissivity;
            Vec3 Diffuse = Vec3(0,0,0);
            Vec3 Ks = Vec3(0,0,0);
            
            for (uint v = 0; v < Samples; v++) {
                Vec3 NewDiffuseSample = DiffuseRaySample(N,state);

                Vec3 H = (V + NewDiffuseSample).unitVector();
                Vec3 Ks = Fresnel(mat.F0, V, H);
                Vec3 Kd = Vec3(1.0f,1.0f,1.0f) - Ks;

                Vec3 cookTorrenceNum = Ks * NormalDist(mat.Roughness, N, H) * G(mat.Roughness, N, V, NewDiffuseSample);
                float cookTorrenceDen = max(4.0f * max(N.dot(V), 0.0f) * max(N.dot(NewDiffuseSample), 0.0f),EPSILON);
                Vec3 cookTorrence = cookTorrenceNum / cookTorrenceDen;

                Ray DiffuseRay = Ray(Pos, NewDiffuseSample);
                Light = Light + mat.Emissivity + (mat.Diffuse * Kd + cookTorrence) * PBR(DiffuseRay, tree, tris, triIndexes, state);
            }
            Light = Light / Samples;


            //Ray SpecularRay = Ray(Pos, -RenderRay.getDirection() + N * N.dot(RenderRay.getDirection()) * 2);
            return Light;
        }
    } else {
        return Vec3(0,0,0); // SKYBOX
    }
    
} */

__global__ void rayTraceKernel(CameraViewport* View, BVHNode* tree, Triangle* tris, uint* triIndexes, PointLight* lights, int lightCount, Vec3 CFramePos, int MaxDist, Vec3 ambience, curandState *d_randStates, uint skyWidth, uint skyHeight, Vec3* skyColors, Vec3* d_Light) {
    uint X = blockIdx.x * blockDim.x + threadIdx.x;
    uint Y = blockIdx.y * blockDim.y + threadIdx.y;

    uint screenWidth = View->Width;
    uint screenHeight = View->Height;

    uint index = X + screenWidth*Y;

    if (X >= screenWidth || Y >= screenHeight) return;

    Vec3 newPos = View->Origin.getPos() + View->unitRight * (__fdividef(2 * (X + 0.5), screenWidth) - 1) + View->unitUp * (1 - __fdividef(2 * (Y + 0.5), screenHeight)) + View->Origin.getLookVector();
    Ray RenderRay = Ray(View->Origin.getPos(), (newPos - View->Origin.getPos()).unitVector());

    IntersectBVH(RenderRay, tree, tris, triIndexes);
    if (RenderRay.hasHit()) {
        #if SAMPLES == 0
            Triangle& hitTri = tris[RenderRay.hit.hit];
            const Material& mat = hitTri.getMaterial();
            d_Light[index] = mat.Emissivity;
        #else
            Ray OriginalRay = RenderRay;
            for (uint q = 0; q < SAMPLES; q++) {
                Vec3 a = Vec3(1,1,1);
                Vec3 V,Pos,NewNorm,H,Ks,Kd,Result;
                RenderRay = OriginalRay;
                
                for (uint i = 0; i < MAXBOUNCES; i++) {
                    V = RenderRay.getDirection();
                    Triangle& hitTri = tris[RenderRay.hit.hit];
                    Pos = RenderRay.getPos() + RenderRay.getDirection() * RenderRay.hit.t;

                    const Material& mat = hitTri.getMaterial();
                    const Vec3& N = hitTri.getNorm();
                    
                    NewNorm = DiffuseSample(d_randStates[index],hitTri);
                    
                    Ks = Fresnel(mat.F0, -V, NewNorm);
                    Kd = Vec3(1.0f,1.0f,1.0f) - Ks;

                    RenderRay = Ray(Pos, NewNorm);
                    Result = mat.Emissivity * a;
                    a = (mat.Diffuse * Kd);
                    IntersectBVH(RenderRay, tree, tris, triIndexes);
                    if (!RenderRay.hasHit()) {
                        atomicAddVec3(d_Light[index], (Result + a * AMBIENT)/SAMPLES);
                        break;
                    } else {
                        atomicAddVec3(d_Light[index], Result/SAMPLES);
                    }
                }

                RenderRay = OriginalRay;
                for (uint i = 0; i < 1; i++) {
                    V = RenderRay.getDirection();
                    Triangle& hitTri = tris[RenderRay.hit.hit];
                    Pos = RenderRay.getPos() + RenderRay.getDirection() * RenderRay.hit.t;

                    const Material& mat = hitTri.getMaterial();
                    const Vec3& N = hitTri.getNorm();
                    
                    NewNorm = SpecSample(V,d_randStates[index],hitTri);
                    
                    Ks = Fresnel(mat.F0, -V, NewNorm);

                    RenderRay = Ray(Pos, -RenderRay.getDirection() + N * N.dot(RenderRay.getDirection()) * 2);
                    a = (Ks * (GGXShadow(mat.Roughness,N,-V,-RenderRay.getDirection())/GGXMask(mat.Roughness,N,-RenderRay.getDirection())));
                    Result = mat.F0 * a;
                    IntersectBVH(RenderRay, tree, tris, triIndexes);
                    if (!RenderRay.hasHit()) {
                        atomicAddVec3(d_Light[index], (Result * getSkyColor(RenderRay.Dir,skyWidth,skyHeight,skyColors))/SAMPLES);
                        break;
                    } else {
                        atomicAddVec3(d_Light[index], (Result * mat.Emissivity)/SAMPLES);
                    }
                }
            }
        
        #endif
    } else {
        d_Light[index] = getSkyColor(RenderRay.Dir,skyWidth,skyHeight,skyColors);
    }
}

__global__ void WriteScreen(CameraViewport* View, uint32_t* screenBuffer, Vec3* d_Light) {
    uint X = blockIdx.x * blockDim.x + threadIdx.x;
    uint Y = blockIdx.y * blockDim.y + threadIdx.y;

    uint screenWidth = View->Width;
    uint screenHeight = View->Height;

    uint index = X + screenWidth*Y;

    if (X >= screenWidth || Y >= screenHeight) return;

    screenBuffer[index] = (d_Light[index]*255).toUint32();
}


void Camera::raytrace(Viewport& screen) {
    int bufferSize = this->View.Width * this->View.Height;
    uint32_t* screenBuffer;
    cudaError_t cudaStatus;

    // Allocate memory for the screen buffer on the device
    cudaStatus = cudaMalloc(&screenBuffer, bufferSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate screenBuffer: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // TODO: Make it so it doesn't do this every frame
    // Allocate memory for the Sample Results
    Vec3* SampleResults = new Vec3[bufferSize];
    Vec3* d_SampleResults;
    cudaStatus = cudaMalloc(&d_SampleResults, bufferSize * sizeof(Vec3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate d_SampleResults: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // Copy the initialized Sample Results from host to device
    cudaStatus = cudaMemcpy(d_SampleResults, SampleResults, bufferSize * sizeof(Vec3), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy SampleResults to device: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // Allocate memory for the CameraViewport on the device
    CameraViewport* CamView;
    cudaStatus = cudaMalloc(&CamView, sizeof(CameraViewport));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate CamView: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // Copy the CameraViewport structure from host to device
    cudaStatus = cudaMemcpy(CamView, &(this->View), sizeof(CameraViewport), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy CameraViewport to device: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    dim3 Grid(ceil((1.0 * this->View.Width) / 32), ceil((1.0 * this->View.Height) / 16), 1);
    dim3 Blocks(32, 16, 1);

    rayTraceKernel<<<Grid, Blocks>>>(CamView, this->world->BVHNodes, this->world->Tris, this->world->TriIndexes, this->world->PointLights, this->world->LightCount, this->CFrame.getPos(), MAXDIST, AMBIENT, this->d_randStates, this->world->sky.width,this->world->sky.height, this->world->sky.deviceColors,d_SampleResults);                                                              

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Synchronize Problem occured: " << cudaGetErrorString(err) << "\n";
    }

    WriteScreen<<<Grid, Blocks>>>(CamView, screenBuffer, d_SampleResults);     
    err = cudaDeviceSynchronize();                                                         
    if (err != cudaSuccess) {
        std::cout << "Synchronize Problem occured during write: " << cudaGetErrorString(err) << "\n";
    }

    err = cudaMemcpy(screen.lockAndGetPixels(),screenBuffer,bufferSize * sizeof(uint32_t),cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cout << "MemCpy Problem occured: " << cudaGetErrorString(err) << "\n";
    }

    screen.setDisplay();
    screen.Update();

    cudaFree(screenBuffer);
    cudaFree(CamView);
    cudaFree(d_SampleResults);
    delete[] SampleResults;
}

CameraViewport::~CameraViewport() {
    
}

Camera::~Camera() {
    std::cout << "Deleting Camera" << "\n";
    cudaFree(this->d_randStates);
}
