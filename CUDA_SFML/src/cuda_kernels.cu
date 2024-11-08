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
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x index of cell
    int y = blockIdx.y * blockDim.y + threadIdx.y; // y index of cell

    if (x >= gridWidth || y >= gridHeight)
        return; // Boundary check

    int neighbors = 0;

    // Count neighbors of the current cell
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < gridWidth && ny >= 0 && ny < gridHeight &&
                !(dx == 0 && dy == 0))
            {
                neighbors += gridCurrent[nx + ny * gridWidth];
            }
        }
    }

    // Conway's Game of Life rules
    int idx = x + y * gridWidth;
    if (gridCurrent[idx])
    {
        gridNext[idx] =
            (neighbors == 2 || neighbors == 3); // Cell remains alive
    }
    else
    {
        gridNext[idx] = (neighbors == 3); // Cell becomes alive
    }
}

void normalMemSimulate(RenderWindow& window, int threadsPerBlock,
                       vector<vector<bool>>& gridCurrent,
                       vector<vector<bool>>& gridNext, int gridWidth,
                       int gridHeight, int cellSize)
//    TODO: Might NOT neet gridWidth and gridHeight
{
    // Initialize grid states on host
    bool *d_gridCurrent, *d_gridNext;

    // Allocate memory on device (GPU)
    cudaMalloc((void**)&d_gridCurrent, gridWidth * gridHeight * sizeof(bool));
    cudaMalloc((void**)&d_gridNext, gridWidth * gridHeight * sizeof(bool));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_gridCurrent, gridCurrent.data(),
               gridWidth * gridHeight * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridNext, gridNext.data(),
               gridWidth * gridHeight * sizeof(bool), cudaMemcpyHostToDevice);

    // // Define block size (32 threads per block)
    // dim3 blockDim(threadsPerBlock, 1); // 32 threads in 1D (x-direction)
    // dim3 gridDim((gridWidth + blockDim.x - 1) / blockDim.x,
    //              (gridHeight + blockDim.y - 1) /
    //                  blockDim.y); // Grid size to cover all cells

    dim3 blockDim(16, 16); // 16x16 threads per block (2D block)
    dim3 gridDim((gridWidth + blockDim.x - 1) / blockDim.x,
                 (gridHeight + blockDim.y - 1) /
                     blockDim.y); // Grid size to cover all cells

    // Run the simulation for multiple generations
    for (int generationCount = 0; window.isOpen(); ++generationCount)
    {
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape))
            {
                window.close();
            }
        }

        // Launch CUDA kernel to update the grid
        updateGridKernel<<<gridDim, blockDim>>>(d_gridCurrent, d_gridNext,
                                                gridWidth, gridHeight);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA kernel launch failed: "
                      << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }

        // Copy the updated grid back to host
        cudaMemcpy(gridCurrent.data(), d_gridNext,
                   gridWidth * gridHeight * sizeof(bool),
                   cudaMemcpyDeviceToHost);

        window.clear();

        // Draw the grid
        for (int x = 0; x < gridWidth; ++x)
        {
            for (int y = 0; y < gridHeight; ++y)
            {
                if (gridCurrent[x][y])
                {
                    RectangleShape cell(Vector2f(cellSize, cellSize));
                    cell.setPosition(x * cellSize, y * cellSize);
                    cell.setFillColor(Color::White);
                    window.draw(cell);
                }
            }
        }

        window.display();

        // Swap grids for the next generation
        gridCurrent = gridNext;

        // Check for performance every 100 generations
        if (generationCount % 100 == 0)
        {
            cout << "Generation " << generationCount << " complete." << endl;
        }
    }

    // Free device memory
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
