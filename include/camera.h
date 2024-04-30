#include "quaternion.h"
#include "world.h"
#include "BVH.h"
#include "viewport.h"

#pragma once

class CameraViewport {
    public:
        Quaternion Origin;
        int Width, Height;
        Vec3 unitRight,unitUp;
        float* AnglePowerMap;

        CameraViewport(Quaternion Origin,int Width, int Height, float FOV, int fromOrigin);
        void SetView(Quaternion Origin,int Width, int Height, float FOV, int fromOrigin);
        ~CameraViewport();
};

class Camera {
    private:
        Quaternion CFrame;
    public:
        int ViewWidth, ViewHeight, fromOrigin;
        float FOV;
        CameraViewport View;
        World* world;

        Camera(World* world, Vec3 Pos,int ViewWidth,int ViewHeight,int fromOrigin, float FOV);
        Camera(World* world, Quaternion Coords,int ViewWidth,int ViewHeight,int fromOrigin, float FOV);
        Vec3 getPos();
        void setCFrame(Quaternion Pos);
        Quaternion& getCFrame();
        Quaternion& rotate(const Vec3& other, double deg);
        Quaternion& move(const Vec3& other);
        void resetView();
        void raytrace(Viewport& screen);
};