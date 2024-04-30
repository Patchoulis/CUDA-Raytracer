#pragma once

#include "Vec3.h"


class Material {
    public:
        Vec3 Albedo;
        Vec3 Emissivity;
        float Roughness;
        Vec3 F0;
        Material(Vec3 Albedo = Vec3(0,0,0), Vec3 Emissivity = Vec3(0,0,0), float Roughness = 0.05, Vec3 F0 = Vec3(0.02,0.02,0.02));
};