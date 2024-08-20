# Compiler settings
CC = g++
NVCC = nvcc

# OpenCV flags
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`

# Existing flags
LDFLAGS = -lSDL2 -lcudart $(OPENCV_FLAGS)
CUDA_VERSION=12.4
INC_DIRS=/usr/local/cuda-${CUDA_VERSION}/include
INC=$(foreach d, $(INC_DIRS), -I$d $(shell pkg-config --cflags opencv4))

CUDAFLAGS = -std=c++11 $(INC)
CXXFLAGS = -std=c++11 $(INC)

# Object files list including skybox.o
OBJS = Engine.o viewport.o quaternion.o Vec3.o camera.o ray.o object.o triangle.o world.o user.o material.o pointlight.o BVH.o skybox.o
EXECUTABLE = Engine

# Default target
default: $(EXECUTABLE)

# Rule for CUDA object files (.cu)
%.o: %.cu %.h
	$(NVCC) $(CUDAFLAGS) -dc -o $@ $<

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -dc -o $@ $<

# Rule for C++ object files (.cpp)
%.o: %.cpp %.h
	$(CC) $(CXXFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CC) $(CXXFLAGS) -c -o $@ $<

# Device linking step
device_link.o: $(OBJS)
	$(NVCC) -dlink $(OBJS) -o device_link.o -lcudart

# Linking the final executable
$(EXECUTABLE): $(OBJS) device_link.o
	$(NVCC) -o $(EXECUTABLE) $(OBJS) device_link.o $(LDFLAGS)

# Clean rule
clean:
	-rm -f $(OBJS) device_link.o $(EXECUTABLE)
