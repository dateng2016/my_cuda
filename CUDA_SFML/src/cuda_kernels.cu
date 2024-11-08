// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include "utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace sf;

// CUDA kernel for updating the grid
__global__ void updateGridKernel(uint8_t* gridCurrent, uint8_t* gridNext,
                                 int gridWidth, int gridHeight)
{
    // ! This is 1D, so we need to unpack it back to (x, y) coordinates
    int l = blockIdx.x * blockDim.x + threadIdx.x; // x index of cell
    int y = l / gridWidth;
    int x = l % gridWidth;

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
    if (gridCurrent[l])
    {
        gridNext[l] = (neighbors == 2 || neighbors == 3); // Cell remains alive
    }
    else
    {
        gridNext[l] = (neighbors == 3); // Cell becomes alive
    }
}

void normalMemSimulate(RenderWindow& window, int threadsPerBlock,
                       vector<vector<uint8_t>>& gridCurrent,
                       vector<vector<uint8_t>>& gridNext, int gridWidth,
                       int gridHeight, int cellSize, string memoryType)
{
    uint8_t *d_gridCurrent, *d_gridNext;
    int N = gridWidth * gridHeight;
    size_t size = N * sizeof(uint8_t);
    // * Allocate Memory on GPU
    if (memoryType == "NORMAL")
    {
        cudaMalloc(&d_gridCurrent, size);
        cudaMalloc(&d_gridNext, size);
    }
    else if (memoryType == "PINNED")
    {
        cudaMallocHost(&d_gridCurrent, size);
        cudaMallocHost(&d_gridNext, size);
    }
    else if (memoryType == "MANAGED")
    {
        cudaMallocManaged(&d_gridCurrent, size);
        cudaMallocManaged(&d_gridNext, size);
    }

    // * Flatten the vectors
    vector<uint8_t> flatGridCurrent;
    vector<uint8_t> flatGridNext;
    flatGridCurrent.reserve(gridWidth *
                            gridHeight); // Reserve memory for efficiency
    flatGridNext.reserve(gridWidth *
                         gridHeight); // Reserve memory for efficiency

    for (int y = 0; y < gridHeight; ++y)
    {
        for (int x = 0; x < gridWidth; ++x)
        {
            flatGridCurrent.push_back(static_cast<uint8_t>(gridCurrent[y][x]));
            flatGridNext.push_back(static_cast<uint8_t>(gridNext[y][x]));
        }
    }
    // * Copy vectors from host to device
    cudaMemcpy(d_gridCurrent, flatGridCurrent.data(), size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridNext, flatGridNext.data(), size, cudaMemcpyHostToDevice);

    // * Determine the number of blocks per grid.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // * Start the simulation
    while (window.isOpen())
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
        updateGridKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_gridCurrent, d_gridNext, gridWidth, gridHeight);

        cudaDeviceSynchronize();

        // * We move the memory from GPU to host to render the image
        cudaMemcpy(flatGridCurrent.data(), d_gridNext, size,
                   cudaMemcpyDeviceToHost);
        // * Start Rendering

        window.clear();

        RectangleShape cell(Vector2f(cellSize, cellSize));

        for (int y = 0; y < gridHeight; ++y)
        {
            for (int x = 0; x < gridWidth; ++x)
            {
                if (flatGridCurrent[y * gridWidth + x])
                {
                    // cell.setPosition(y * cellSize, x * cellSize);
                    cell.setPosition(x * cellSize, y * cellSize);
                    cell.setFillColor(Color::White);
                    window.draw(cell);
                }
            }
        }

        window.display();

        // * We do the memory swap INSIDE the GPU so we do not have to
        // * move the memory from HOST to GPU AGAIN.

        uint8_t* temp = d_gridCurrent;
        d_gridCurrent = d_gridNext;
        d_gridNext = temp;
        // cudaMemcpy(d_gridCurrent, flatGridCurrent.data(), size,
        //            cudaMemcpyHostToDevice);
    }

    // * Free the GPU Memory when the while loop finishes
    cudaFree(d_gridCurrent);
    cudaFree(d_gridNext);
}
