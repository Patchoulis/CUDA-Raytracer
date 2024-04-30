#pragma once

#include "camera.h"
#include "viewport.h"

#define YPanL SDL_SCANCODE_LEFT
#define YPanR SDL_SCANCODE_RIGHT

#define XPanU SDL_SCANCODE_UP
#define XPanD SDL_SCANCODE_DOWN

#define ZPanL SDL_SCANCODE_Q
#define ZPanR SDL_SCANCODE_E

#define MoveUp SDL_SCANCODE_R
#define MoveDown SDL_SCANCODE_F

#define MoveLeft SDL_SCANCODE_A
#define MoveRight SDL_SCANCODE_D

#define MoveForward SDL_SCANCODE_W
#define MoveBackward SDL_SCANCODE_S

#define HalfDegree 0.00027270769

struct Configs {
    SDL_Scancode MoveUpKey;
    SDL_Scancode MoveDownKey;

    SDL_Scancode MoveForwardKey;
    SDL_Scancode MoveBackwardKey;

    SDL_Scancode MoveLeftKey;
    SDL_Scancode MoveRightKey;

    SDL_Scancode YPanLKey;
    SDL_Scancode YPanRKey;

    SDL_Scancode XPanLKey;
    SDL_Scancode XPanRKey;

    SDL_Scancode ZPanLKey;
    SDL_Scancode ZPanRKey;

    float MoveSpeed;
    double RotateSpeed;
};

class User {
    private:
        Camera cam;
        Configs config;
        Viewport screen;
    public:
        User(Camera cam, Viewport screen, Configs config = Configs{ MoveUp, MoveDown, MoveForward, MoveBackward, MoveLeft, MoveRight, YPanL, YPanR, XPanU, XPanD, ZPanL, ZPanR, 0.000005, HalfDegree/300});
        void ProcessMovement();
        void setConfigs(Configs config);
        const Configs& getConfigs() const;
        Camera& getCamera();
        void setCamera(Camera cam);
        void raytrace();
};