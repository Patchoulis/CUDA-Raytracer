#include "user.h"

User::User(Camera &cam, Viewport screen, Configs config) : cam(cam), config(config), screen(screen) {}

void User::ProcessMovement() {
    const Uint8* keystate = SDL_GetKeyboardState(NULL);
    
    if(keystate[this->config.YPanLKey])
    {
        this->cam.rotate(this->cam.getCFrame().getUpVector(), this->config.RotateSpeed);
    } 
    else if(keystate[this->config.YPanRKey]) 
    {
        this->cam.rotate(this->cam.getCFrame().getUpVector(), -this->config.RotateSpeed);
    }
    if(keystate[this->config.XPanLKey])
    {
        this->cam.rotate(this->cam.getCFrame().getRightVector(), this->config.RotateSpeed);
    }
    else if(keystate[this->config.XPanRKey])
    {
        this->cam.rotate(this->cam.getCFrame().getRightVector(), -this->config.RotateSpeed);
    }
    if(keystate[this->config.ZPanLKey])
    {
        this->cam.rotate(this->cam.getCFrame().getLookVector(), -this->config.RotateSpeed);
    }
    else if(keystate[this->config.ZPanRKey])
    {
        this->cam.rotate(this->cam.getCFrame().getLookVector(), this->config.RotateSpeed);
    }


    if(keystate[this->config.MoveUpKey])
    {
        this->cam.move(this->cam.getCFrame().getUpVector() * this->config.MoveSpeed);
    }
    if(keystate[this->config.MoveDownKey])
    {
        this->cam.move(-this->cam.getCFrame().getUpVector() * this->config.MoveSpeed);
    }
    if(keystate[this->config.MoveRightKey])
    {
        this->cam.move(this->cam.getCFrame().getRightVector() * this->config.MoveSpeed);
    }
    if(keystate[this->config.MoveLeftKey])
    {
        this->cam.move(-this->cam.getCFrame().getRightVector() * this->config.MoveSpeed);
    }
    if(keystate[this->config.MoveForwardKey])
    {
        this->cam.move(this->cam.getCFrame().getLookVector() * this->config.MoveSpeed);
    }
    if(keystate[this->config.MoveBackwardKey])
    {
        this->cam.move(-this->cam.getCFrame().getLookVector() * this->config.MoveSpeed);
    }
    if(keystate[SDL_SCANCODE_C])
    {
        std::cout << "USER: " << this->cam.View.Origin << "\n";
    }

}

void User::setConfigs(Configs config) {
    this->config = config;
}

const Configs& User::getConfigs() const {
    return this->config;
}

Camera& User::getCamera() {
    return this->cam;
}

void User::setCamera(Camera &cam) {
    this->cam = cam;
}

void User::raytrace() {
    this->cam.raytrace(this->screen);
}