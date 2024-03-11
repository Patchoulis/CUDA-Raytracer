#include "viewport.h"
#include <iostream>


Viewport::Viewport(int WindowX, int WindowY, const char* Name) : maxX(WindowX), maxY(WindowY) {
    SDL_Window* window = SDL_CreateWindow(Name,
                                      SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                      WindowX, WindowY,
                                      SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    if (window == NULL) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
    }
    this->window = window;

    this->screen = SDL_CreateRGBSurface( 0, WindowX, WindowY, 32, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF);
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    this->renderer = renderer;

    SDL_Texture* windowTexture = SDL_CreateTexture(this->renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET, this->maxX, this->maxY);
    if (!windowTexture) {
        std::cerr << "Window Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    this->texture = windowTexture;

    this->maxX = WindowX;
    this->maxY = WindowY;
}

void Viewport::Resize(int X, int Y) {
    this->maxX = X;
    this->maxY = Y;
}

void Viewport::drawPixel(int x, int y, SDL_Color color) {
    SDL_SetRenderDrawColor(this->renderer, color.r, color.g, color.b, color.a);
    SDL_RenderDrawPoint(this->renderer, x, y);
}

void* Viewport::lockAndGetPixels() {
    SDL_LockSurface(this->screen);
    return this->screen->pixels;
}

void Viewport::setDisplay() {
    SDL_UnlockSurface(this->screen);
    SDL_UpdateTexture(this->texture, NULL, this->screen->pixels, this->screen->pitch);
    SDL_RenderClear(this->renderer);
    SDL_RenderCopy(this->renderer, this->texture, NULL, NULL);
}

void Viewport::ClearScreen() {
    SDL_SetRenderDrawColor(this->renderer, 0, 0, 0, 255);
    SDL_RenderClear(this->renderer);
}

void Viewport::Update() {
    SDL_RenderPresent(this->renderer);
}

void Viewport::Quit() {
    SDL_DestroyRenderer(this->renderer);
    SDL_DestroyWindow(this->window);
    SDL_Quit();
}