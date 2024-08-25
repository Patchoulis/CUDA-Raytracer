#include "Vec3.h"
#include <opencv4/opencv2/opencv.hpp>

#pragma once

class Skybox {
    private:
        Vec3* getColorsFromFile(const char*& file);
    public:
        Vec3* Colors;
        Vec3* deviceColors;
        uint height;
        uint width;
        Skybox(const char*& file);
        Vec3* createDeviceColors() const;
        ~Skybox();
};

__device__ Vec3& getSkyColor(Vec3& Dir, uint width, uint height, Vec3*& colors);