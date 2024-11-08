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
                       int gridHeight, int cellSize, string memoryType);
void pinnedMemSimulate(RenderWindow& window, int threadsPerBlock,
                       vector<vector<uint8_t>>& gridCurrent,
                       vector<vector<uint8_t>>& gridNext, int gridWidth,
                       int gridHeight, int cellSize, string memoryType);
void managedMemSimulate(RenderWindow& window, int threadsPerBlock,
                        vector<vector<uint8_t>>& gridCurrent,
                        vector<vector<uint8_t>>& gridNext, int gridWidth,
                        int gridHeight, int cellSize, string memoryType);

#endif // CUDA_KERNELS_H
