#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace sf;

// CUDA kernel for updating the grid
__global__ void updateGridKernel(uint8_t* gridCurrent, uint8_t* gridNext,
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
                       vector<vector<uint8_t>>& gridCurrent,
                       vector<vector<uint8_t>>& gridNext, int gridWidth,
                       int gridHeight, int cellSize)
{
    // Initialize grid states on host
    uint8_t *d_gridCurrent, *d_gridNext;

    // Allocate memory on device (GPU)
    cudaError_t err = cudaMalloc((void**)&d_gridCurrent,
                                 gridWidth * gridHeight * sizeof(uint8_t));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA memory allocation failed: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    cudaMalloc((void**)&d_gridNext, gridWidth * gridHeight * sizeof(uint8_t));

    // Copy data from host (CPU) to device (GPU)
    cudaMemcpy(d_gridCurrent, gridCurrent.data()->data(),
               gridWidth * gridHeight * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridNext, gridNext.data()->data(),
               gridWidth * gridHeight * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    // Define block size (16x16 threads per block)
    dim3 blockDim(16, 16); // 16x16 threads in 2D blocks
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
        cudaMemcpy(gridCurrent.data()->data(), d_gridNext,
                   gridWidth * gridHeight * sizeof(uint8_t),
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
