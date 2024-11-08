// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include "utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace sf;

// CUDA kernel for updating the grid
__global__ void updateGridKernel(bool* gridCurrent, bool* gridNext,
                                 int gridWidth, int gridHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= gridWidth || y >= gridHeight)
        return;

    int neighbors = 0;
    int idx = x + y * gridWidth;

    // Count neighbors of the current cell
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            if (dx == 0 && dy == 0)
                continue;

            int nx = x + dx;
            int ny = y + dy;

            // Wrap around edges
            nx = (nx + gridWidth) % gridWidth;
            ny = (ny + gridHeight) % gridHeight;

            neighbors += gridCurrent[nx + ny * gridWidth];
        }
    }

    // Conway's Game of Life rules
    gridNext[idx] = (neighbors == 3 || (neighbors == 2 && gridCurrent[idx]));
}

void normalMemSimulate(sf::RenderWindow& window, int threadsPerBlock,
                       vector<vector<bool>>& grid, int gridWidth,
                       int gridHeight, int cellSize)
{
    // Create flat arrays for CUDA
    vector<bool> flatGridCurrent(gridWidth * gridHeight);
    vector<bool> flatGridNext(gridWidth * gridHeight);

    // Convert 2D grid to flat array
    for (int y = 0; y < gridHeight; y++)
    {
        for (int x = 0; x < gridWidth; x++)
        {
            flatGridCurrent[x + y * gridWidth] = grid[x][y];
        }
    }

    // Initialize grid states on device
    bool *d_gridCurrent, *d_gridNext;

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_gridCurrent, gridWidth * gridHeight * sizeof(bool));
    cudaMalloc((void**)&d_gridNext, gridWidth * gridHeight * sizeof(bool));

    // Copy initial state to device
    cudaMemcpy(d_gridCurrent, flatGridCurrent.data(),
               gridWidth * gridHeight * sizeof(bool), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim((gridWidth + blockDim.x - 1) / blockDim.x,
                 (gridHeight + blockDim.y - 1) / blockDim.y);

    // Main simulation loop
    for (int generationCount = 0; window.isOpen(); ++generationCount)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed ||
                sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            {
                window.close();
            }
        }

        // Launch CUDA kernel
        updateGridKernel<<<gridDim, blockDim>>>(d_gridCurrent, d_gridNext,
                                                gridWidth, gridHeight);

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA kernel launch failed: "
                      << cudaGetErrorString(err) << std::endl;
            cudaFree(d_gridCurrent);
            cudaFree(d_gridNext);
            exit(EXIT_FAILURE);
        }

        // Synchronize before copying back
        cudaDeviceSynchronize();

        // Copy the updated grid back to host
        cudaMemcpy(flatGridCurrent.data(), d_gridNext,
                   gridWidth * gridHeight * sizeof(bool),
                   cudaMemcpyDeviceToHost);

        // Convert flat array back to 2D grid for rendering
        for (int y = 0; y < gridHeight; y++)
        {
            for (int x = 0; x < gridWidth; x++)
            {
                grid[x][y] = flatGridCurrent[x + y * gridWidth];
            }
        }

        window.clear();

        // Draw the grid
        for (int x = 0; x < gridWidth; ++x)
        {
            for (int y = 0; y < gridHeight; ++y)
            {
                if (grid[x][y])
                {
                    sf::RectangleShape cell(sf::Vector2f(cellSize, cellSize));
                    cell.setPosition(x * cellSize, y * cellSize);
                    cell.setFillColor(sf::Color::White);
                    window.draw(cell);
                }
            }
        }

        window.display();

        // Swap device pointers
        bool* temp = d_gridCurrent;
        d_gridCurrent = d_gridNext;
        d_gridNext = temp;

        if (generationCount % 100 == 0)
        {
            std::cout << "Generation " << generationCount << " complete."
                      << std::endl;
        }
    }

    // Cleanup
    cudaFree(d_gridCurrent);
    cudaFree(d_gridNext);
}
// __global__ void vectorAddKernel(const float* A, const float* B, float* C, int
// N)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N)
//     {
//         C[i] = A[i] + B[i];
//     }
// }

// void vectorAdd(const float* A, const float* B, float* C, int N)
// {
//     float *d_A, *d_B, *d_C;
//     size_t size = N * sizeof(float);

//     // Allocate memory on GPU
//     cudaMalloc(&d_A, size);
//     cudaMalloc(&d_B, size);
//     cudaMalloc(&d_C, size);

//     // Copy vectors from host to device
//     cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

//     // Launch kernel
//     int threadsPerBlock = 32;
//     int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
//     std::cout << "blocksPerGrid = " << blocksPerGrid << std::endl;

//      vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

//     // Check for kernel launch errors
//     cudaError_t err = cudaGetLastError();
//     cout << "Error DETECTION before" << endl;
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err)
//                   << std::endl;
//         // exit(EXIT_FAILURE);
//     }
//     cout << "Error DETECTION AFTER" << endl;

//     cudaDeviceSynchronize();
//     // Copy result back to host
//     cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

//     for (int i = 0; i < 10; ++i)
//     {
//         std::cout << "C[" << i << "] = " << C[i] << std::endl;
//     }

//     // Free memory on GPU
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }
