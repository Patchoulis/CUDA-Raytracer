# This Makefile is for reference

#CC = g++
#NVCC = nvcc
#LDFLAGS = -lSDL2 -lcudart
#
#CUDA_VERSION=12.4
#INC_DIRS=/usr/local/cuda-${CUDA_VERSION}/include include/
#INC=$(foreach d, $(INC_DIRS), -I$d)
#CUDAFLAGS = -dc $(INC) -std=c++11
#
#SRC_DIR = src
#INCLUDE_DIR = include
#
#OBJS = $(SRC_DIR)/Engine.o $(SRC_DIR)/viewport.o $(SRC_DIR)/quaternion.o $(SRC_DIR)/Vec3.o $(SRC_DIR)/camera.o $(SRC_DIR)/ray.o $(SRC_DIR)/object.o $(SRC_DIR)/triangle.o $(SRC_DIR)/world.o $(SRC_DIR)/user.o $(SRC_DIR)/material.o $(SRC_DIR)/pointlight.o $(SRC_DIR)/BVH.o
#EXECUTABLE = Engine
#
#default: $(EXECUTABLE)
#
#$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu $(INCLUDE_DIR)/%.h
#	$(NVCC) $(CUDAFLAGS) -c -o $@ $< $(LDFLAGS)
#
#$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
#	$(NVCC) $(CUDAFLAGS) -c -o $@ $< $(LDFLAGS)
#
#$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp $(INCLUDE_DIR)/%.h
#	$(CC) -c -o $@ $< $(LDFLAGS)
#
#$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
#	$(CC) -c -o $@ $< $(LDFLAGS)
#
#$(EXECUTABLE): $(OBJS)
#	$(NVCC) -dlink $(OBJS) -o $(SRC_DIR)/device_link.o -lcudart
#	$(CC) -o $(EXECUTABLE) -L/usr/local/cuda-12.4/lib64 $(OBJS) $(SRC_DIR)/device_link.o $(LDFLAGS)
#
#clean:
#	-rm $(OBJS) $(SRC_DIR)/device_link.o $(EXECUTABLE)
