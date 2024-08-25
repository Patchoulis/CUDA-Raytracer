#include "Vec3.h"

#pragma once

class Material {
    public:
        Vec3 Emissivity;
        float Roughness;
        float Reflectance;
        float Metallic;
        Vec3 BaseColor;
        Vec3 Diffuse;
        Vec3 F0;
        float ClearCoat;
        float ClearCoatRoughness;
        Material(Vec3 Emissivity = Vec3(0,0,0), float Roughness = 0.05, float Reflectance=0.0f, float Metallic=0.9, Vec3 BaseColor = Vec3(0,0,0),float ClearCoat=0, float ClearCoatRoughness=0);
        Vec3 CalcF0();
        Vec3 CalcDiffuse();
};