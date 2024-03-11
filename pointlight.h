#include "Vec3.h"
#include "color3.h"

class PointLight {
    public:
        Vec3 Pos;
        Color3 Color;
        float intensity;
        PointLight(Vec3 Pos, Color3 color = Color3(255,255,255), float intensity = 1);
};