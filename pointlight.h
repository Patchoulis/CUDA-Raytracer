#include "Vec3.h"

class PointLight {
    public:
        Vec3 Pos;
        Vec3 Color;
        float intensity;
        PointLight(Vec3 Pos, Vec3 color = Vec3(255,255,255), float intensity = 1);
};