#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <SFML/Graphics.hpp> // For RenderWindow (needed for SFML)
#include <vector>

using namespace std;
using namespace sf;

// Function declaration for normalMemSimulate
void normalMemSimulate(RenderWindow& window, int threadsPerBlock,
                       vector<vector<uint8_t>>& gridCurrent,
                       vector<vector<uint8_t>>& gridNext, int gridWidth,
                       int gridHeight, int cellSize, str memoryType);
void vectorAdd(const float* A, const float* B, float* C, int N);

#endif // CUDA_KERNELS_H
