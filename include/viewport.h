#include <SDL2/SDL.h>

#pragma once

class Viewport {
    private:
        int maxY;
        int maxX;
        SDL_Window* window;
        SDL_Renderer* renderer;
        SDL_Surface* screen;
        SDL_Texture* texture;
    public:
        Viewport(int WindowX, int WindowY, const char* Name);
        void ClearScreen();
        void Resize(int X, int Y);
        void Update();
        void drawPixel(int x, int y, SDL_Color color);
        void setDisplay();
        void* lockAndGetPixels();
        void Quit();
};