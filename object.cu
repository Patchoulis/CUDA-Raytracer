#include "object.h"

Object::Object(std::vector<Triangle> Triangles, Quaternion CFrame) : Triangles(Triangles), CFrame(CFrame) {}

bool sysIsLittleEndian() {
    uint16_t number = 0x1;
    char *bytePtr = reinterpret_cast<char*>(&number);
    return bytePtr[0] == 1;
}

void Object::setCFrame(Quaternion CFrame) {
    this->CFrame = CFrame;
}

void Object::setPos(Vec3 Pos) {
    Vec3 Diff = this->CFrame.getPos() - Pos;
    this->CFrame.setPos(Pos);
    for (Triangle& Tri: this->Triangles) {
        Tri.p1 += Diff;
        Tri.p2 += Diff;
        Tri.p3 += Diff;
    }
}

Object makeCuboid(Vec3 Size, Quaternion CFrame, Material material) {
    Vec3 pos = CFrame.getPos();
    Vec3 right = CFrame.getRightVector();
    Vec3 up = CFrame.getUpVector();
    Vec3 look = CFrame.getLookVector();

    Vec3 bfl = pos - look * (-Size.z/2) + right * (-Size.x/2) + up * (-Size.y/2);
    Vec3 bfr = pos - look * (-Size.z/2) + right * (Size.x/2) + up * (-Size.y/2);
    Vec3 tfr = pos - look * (-Size.z/2) + right * (Size.x/2) + up * (Size.y/2);
    Vec3 tfl = pos - look * (-Size.z/2) + right * (-Size.x/2) + up * (Size.y/2);

    Vec3 bbl = pos - look * (Size.z/2) + right * (-Size.x/2) + up * (-Size.y/2);
    Vec3 bbr = pos - look * (Size.z/2) + right * (Size.x/2) + up * (-Size.y/2);
    Vec3 tbr = pos - look * (Size.z/2) + right * (Size.x/2) + up * (Size.y/2);
    Vec3 tbl = pos - look * (Size.z/2) + right * (-Size.x/2) + up * (Size.y/2);

    std::vector<Triangle> SquareTris{ Triangle(bfr,bfl,tfl,material), Triangle(tfr,bfr,tfl,material), Triangle(bbl,bbr,tbl,material), Triangle(tbl,bbr,tbr,material), Triangle(tfl,bfl,bbl,material), Triangle(tbl,tfl,bbl,material), 
        Triangle(bfr,tfr,bbr,material), Triangle(tfr,tbr,bbr,material), Triangle(tfr,tfl,tbl,material), Triangle(tbr,tfr,tbl,material), Triangle(bfl,bfr,bbl,material), Triangle(bbl,bfr,bbr,material) };
    return Object(SquareTris,CFrame);
}

Object makeMesh(const char*& filename,Quaternion CFrame,Vec3 Size,Material material) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Unable to open file");
    }

    char header[80];
    file.read(header, 80);

    uint32_t numTriangles;
    file.read(reinterpret_cast<char*>(&numTriangles), 4);
    std::cout << "NUMBER OF TRIANGLES: " << numTriangles << "\n";
    std::cout << "SYSTEM: " << (sysIsLittleEndian() ? "LITTLE-ENDIAN" : "BIG-ENDIAN") << "\n";

    std::vector<Vec3> norms{};
    std::vector<Vec3> p1s{};
    std::vector<Vec3> p2s{};
    std::vector<Vec3> p3s{};

    float minx,miny,minz = 3.402823E+38;
    float maxx,maxy,maxz = 1.175494351E-38;
    float x,y,z = 0;

    for (uint i = 0; i < numTriangles; i++) {
        float p1x,p1y,p1z,p2x,p2y,p2z,p3x,p3y,p3z,normx,normy,normz;
        file.read(reinterpret_cast<char*>(&normx), sizeof(float));
        file.read(reinterpret_cast<char*>(&normy), sizeof(float));
        file.read(reinterpret_cast<char*>(&normz), sizeof(float));

        file.read(reinterpret_cast<char*>(&p1x), sizeof(float));
        file.read(reinterpret_cast<char*>(&p1y), sizeof(float));
        file.read(reinterpret_cast<char*>(&p1z), sizeof(float));

        file.read(reinterpret_cast<char*>(&p2x), sizeof(float));
        file.read(reinterpret_cast<char*>(&p2y), sizeof(float));
        file.read(reinterpret_cast<char*>(&p2z), sizeof(float));

        file.read(reinterpret_cast<char*>(&p3x), sizeof(float));
        file.read(reinterpret_cast<char*>(&p3y), sizeof(float));
        file.read(reinterpret_cast<char*>(&p3z), sizeof(float));

        minx = min(minx,p1x); minx = min(minx,p2x); minx = min(minx,p3x);
        miny = min(miny,p1y); miny = min(miny,p2y); miny = min(miny,p3y);
        minz = min(minz,p1z); minz = min(minz,p2z); minz = min(minz,p3z);

        maxx = max(maxx,p1x); maxx = max(maxx,p2x); maxx = max(maxx,p3x);
        maxy = max(maxy,p1y); maxy = max(maxy,p2y); maxy = max(maxy,p3y);
        maxz = max(maxz,p1z); maxz = max(maxz,p2z); maxz = max(maxz,p3z);

        p1s.push_back(Vec3(p1x,p1y,p1z));
        p2s.push_back(Vec3(p2x,p2y,p2z));
        p3s.push_back(Vec3(p3x,p3y,p3z));
        norms.push_back(Vec3(normx,normy,normz));
        std::cout << p1s[i] << " " << p2s[i] << " " << p3s[i] << "\n";

        x+= p1x + p2x + p3x;
        y+= p1y + p2y + p3y;
        z+= p1z + p2z + p3z;

        uint16_t attributeByteCount;
        file.read(reinterpret_cast<char*>(&attributeByteCount), 2);
    }

    x /= (numTriangles*3.0f);
    y /= (numTriangles*3.0f);
    z /= (numTriangles*3.0f);

    float sizex = maxx-minx;
    float sizey = maxy-miny;
    float sizez = maxz-minz;
    Vec3 Center = Vec3(x,y,z);

    Vec3 pos = CFrame.getPos();
    Vec3 right = CFrame.getRightVector();
    Vec3 up = CFrame.getUpVector();
    Vec3 look = CFrame.getLookVector();

    std::vector<Triangle> MeshTris{};

    for (uint i = 0; i < numTriangles; i++) {
        Vec3 diff1 = p1s[i]-Center;
        Vec3 diff2 = p2s[i]-Center;
        Vec3 diff3 = p3s[i]-Center;

        float scaleX = sizex != 0 ? Size.x/sizex : 0;
        float scaleY = sizey != 0 ? Size.y/sizey : 0;
        float scaleZ = sizez != 0 ? Size.z/sizez : 0;

        float scaleInvX = scaleX != 0 ? 1/scaleX : 0;
        float scaleInvY = scaleY != 0 ? 1/scaleY : 0;
        float scaleInvZ = scaleZ != 0 ? 1/scaleZ : 0;
        
        p1s[i] = pos + right * (diff1.x * scaleX) + up * (diff1.y * scaleY) - look * (diff1.z * scaleZ);
        p2s[i] = pos + right * (diff2.x * scaleX) + up * (diff2.y * scaleY) - look * (diff2.z * scaleZ);
        p3s[i] = pos + right * (diff3.x * scaleX) + up * (diff3.y * scaleY) - look * (diff3.z * scaleZ);

        norms[i] = (right * norms[i].x * scaleInvX + up * norms[i].y * scaleInvY - look * norms[i].z * scaleInvZ).unitVector();
        //std::cout << p1s[i] << " " << p2s[i] << " " << p3s[i] << "\n";
        MeshTris.push_back(Triangle(p1s[i],p2s[i],p3s[i],norms[i],material));
    }
    return Object(MeshTris,CFrame);
}