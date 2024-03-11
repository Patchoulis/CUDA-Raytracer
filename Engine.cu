#include <SDL2/SDL.h>
#include <iostream>
#include "viewport.h"
#include "user.h"
#include "world.h"
#include <vector>

const int FPS = 30;
const int FRAME_DELAY = 1000 / FPS;

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }
    Viewport screen = Viewport(1000,1000,"Engine");
    
    Object Cuboid = makeCuboid(Vec3(3,3,3),Quaternion(Vec3(0,0,-5)));
    Object BasePlate = makeCuboid(Vec3(25,1,25),Quaternion(Vec3(0,-2,0)));
    World world = World(std::vector<Object> {Cuboid, BasePlate});
    PointLight light = PointLight(Vec3(10,10,10));
    PointLight light2 = PointLight(Vec3(10,10,6));

    world.AddPointLight(light);
    world.AddPointLight(light2);


    Camera cam = Camera(&world, Quaternion(Vec3(0,0,0)),1000,1000,2, M_PI/4);
    User user = User(cam, screen);

    bool quit = false;
    SDL_Event e;

    screen.ClearScreen();
    screen.Update();


    Uint32 frameStart = 0;
    
    while (!quit) {

        user.ProcessMovement();
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            else if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED) {
                screen.Resize(e.window.data1,e.window.data2);
            }
        }
        Uint32 TimePassed = SDL_GetTicks() - frameStart;
        if (TimePassed >= FRAME_DELAY) {
            user.raytrace();
            
            frameStart = SDL_GetTicks();
        }
    }

    screen.Quit();

    return 0;
}
