#include "color3.h"

class Material {
    public:
        Color3 color;
        float Specularity;
        float Refractivity;
        float Reflectivity;
        float Transparency;
        Material(Color3 color, float Specularity = 0, float Refractivity = 0,float Reflectivity = 0, float Transparency = 0);
        __host__ __device__ void setColor(uint8_t r, uint8_t g, uint8_t b);
        __host__ __device__ void setColor(Color3 color);
        __host__ __device__ const Color3& getColor() const;
};