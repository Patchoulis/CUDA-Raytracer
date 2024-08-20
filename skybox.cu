#include "skybox.h"

Vec3* Skybox::getColorsFromFile(const char*& file) {
    cv::Mat Image = cv::imread(file, cv::IMREAD_UNCHANGED);
    std::cout << "IMAGE TYPE: " << cv::typeToString(Image.type()) << "\n";

    if (Image.empty()) {
        std::cout << "Could not open or find the image!" << "\n";
        this->height = 1;
        this->width = 1;
        Vec3* Colors = static_cast<Vec3*>(operator new[](sizeof(Vec3)));
        new (Colors) Vec3(Vec3(0,0,0));
        return Colors;
    } else {
        this->height = Image.rows;
        this->width = Image.cols;
        Vec3* Colors = static_cast<Vec3*>(operator new[](this->width*this->height * sizeof(Vec3)));

        for (int y = 0; y < this->height; y++) {
            for (int x = 0; x < this->width; x++) {
                cv::Vec4f pixel = Image.at<cv::Vec4f>(y, x);
                uint ind = x + (this->height-1-y) * this->width;
                new (Colors + ind) Vec3(Vec3(pixel[2],pixel[1],pixel[0]));
            }
        }
        return Colors;
    }
}

Vec3* Skybox::createDeviceColors() const {
    Vec3* deviceColors = nullptr;
    cudaMalloc((void**) &deviceColors, this->width * this->height * sizeof(Vec3));
    cudaMemcpy(deviceColors, this->Colors, this->width * this->height * sizeof(Vec3), cudaMemcpyHostToDevice);

    return deviceColors;
}

Skybox::Skybox(const char*& file) : Colors(this->getColorsFromFile(file)), deviceColors(this->createDeviceColors()) {}


__device__ Vec3& getSkyColor(Vec3& Dir, uint width, uint height, Vec3*& colors) {
    Vec3 unitDir = Dir.unitVector();
    uint y = round((unitDir.getY()+1) * 0.5 * (height-1));
    float unitX = unitDir.getX();
    float unitZ = unitDir.getZ();
    uint x = round(((atan2(unitZ,unitX)*M_1_PI*0.5+0.5))*(width-1));
    return colors[x + y * width];
}


Skybox::~Skybox() {
    std::cout << "FREED SKYBOX\n";
    for (uint i = 0; i < this->width * this->height; i++) {
        this->Colors[i].~Vec3();
    }
    operator delete[](this->Colors);
    cudaFree(this->deviceColors);
}