#include "utils.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace sf;

// cuda_kernels.cuh
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// void vectorAdd(const float* A, const float* B, float* C, int N);
void normalMemSimulate(RenderWindow& window, int threadsPerBlock,
                       vector<vector<bool>>& gridCurrent,
                       vector<vector<bool>>& gridNext, int gridWidth,
                       int gridHeight);

#endif
