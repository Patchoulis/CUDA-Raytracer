#include "material.h"
//F0(Vec3(((RefractiveIndex-1) * (RefractiveIndex-1))/((RefractiveIndex+1) * (RefractiveIndex+1)),((RefractiveIndex-1) * (RefractiveIndex-1))/((RefractiveIndex+1) * (RefractiveIndex+1)),((RefractiveIndex-1) * (RefractiveIndex-1))/((RefractiveIndex+1) * (RefractiveIndex+1))))


Material::Material(Vec3 Emissivity, float Roughness, float Reflectance, float Metallic,Vec3 BaseColor,float ClearCoat, float ClearCoatRoughness) : Emissivity(Emissivity), Roughness(Roughness), 
    Reflectance(Reflectance), Metallic(Metallic), BaseColor(BaseColor), Diffuse(this->CalcDiffuse()), F0(this->CalcF0()), ClearCoat(ClearCoat), ClearCoatRoughness(ClearCoatRoughness) {}

Vec3 Material::CalcF0() {
    return (this->BaseColor * this->Metallic) + (Vec3(0.16,0.16,0.16) * (1 - this->Metallic) * this->Reflectance * this->Reflectance);
}

Vec3 Material::CalcDiffuse() {
    return this->BaseColor * (1 - this->Metallic);
}