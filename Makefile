CC = g++
NVCC = nvcc
LDFLAGS = -lSDL2 -lcudart

CUDA_VERSION=12.4
INC_DIRS=/usr/local/cuda-${CUDA_VERSION}/include
INC=$(foreach d, $(INC_DIRS), -I$d)
CUDAFLAGS = -dc $(INC) -std=c++11

OBJS = Engine.o viewport.o quaternion.o Vec3.o camera.o ray.o object.o triangle.o world.o user.o material.o pointlight.o BVH.o
EXECUTABLE = Engine

default: $(EXECUTABLE)

%.o: %.cu %.h
	$(NVCC) $(CUDAFLAGS) -c -o $@ $< $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -c -o $@ $< $(LDFLAGS)

%.o: %.cpp %.h
	$(CC) -c -o $@ $< $(LDFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(LDFLAGS)

$(EXECUTABLE): $(OBJS)
	$(NVCC) -dlink $(OBJS) -o device_link.o -lcudart
	$(CC) -o $(EXECUTABLE) $(OBJS) device_link.o $(LDFLAGS)


clean:
	-rm $(OBJS) device_link.o $(EXECUTABLE)