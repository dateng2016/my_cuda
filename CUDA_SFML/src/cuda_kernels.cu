/*
Author: Da Teng
Class: ECE6122
Last Date Modified: date 11/8/2024
Description:
This file gives the kernel function needed to compute for the game of life video
game
*/

// cuda_kernels.cu
#include "cuda_kernels.cuh"
#include "utils.h"
#include <chrono>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace sf;

// CUDA kernel for updating the grid
__global__ void updateGridKernel(uint8_t* gridCurrent, uint8_t* gridNext,
                                 int gridWidth, int gridHeight)
{
    /**
     * @brief CUDA kernel to update the grid state in Conway's Game of Life.
     *
     * This kernel computes the next state of each cell in the grid based on the
     * current state and the number of live neighbors around it. The computation
     * follows the rules of Conway's Game of Life, where:
     * - A live cell with 2 or 3 live neighbors remains alive.
     * - A dead cell with exactly 3 live neighbors becomes alive.
     * - All other cells die or stay dead.
     *
     * The kernel operates on a 1D grid representation (flattened from 2D) and
     * updates the state in the corresponding position in the next grid
     * (gridNext).
     *
     * @param gridCurrent Pointer to the current grid state, a 1D array
     * representing the grid of cells. Each element is either 0 (dead) or 1
     * (alive).
     * @param gridNext Pointer to the next grid state, where the updated values
     * will be written. It is the same size as gridCurrent.
     * @param gridWidth The width of the grid (number of columns).
     * @param gridHeight The height of the grid (number of rows).
     *
     * @return void This kernel does not return any value directly. It modifies
     * the `gridNext` array in place.

     */

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
    /**
     * @brief Simulates Conway's Game of Life using normal CUDA memory
     * allocation.
     *
     * This function simulates the evolution of Conway's Game of Life by
     * repeatedly updating the grid based on the current state of the cells and
     * their neighbors. The simulation runs in parallel on the GPU using CUDA,
     * and the grid is stored in device memory allocated with `cudaMalloc`. The
     * grid is rendered in a window using the SFML library, with each generation
     * being displayed on the screen.
     *
     * The function measures the processing time for every 100 generations and
     * prints the time taken to the console, excluding the time spent rendering
     * the grid.
     *
     * @param window A reference to an SFML `RenderWindow` object used for
     * rendering the grid to the screen.
     * @param threadsPerBlock The number of threads to use per CUDA block for
     * kernel execution.
     * @param gridCurrent A 2D vector of `uint8_t` representing the current
     * state of the grid, where each element is either 0 (dead) or 1 (alive).
     * @param gridNext A 2D vector of `uint8_t` where the updated state of the
     * grid will be stored.
     * @param gridWidth The width (number of columns) of the grid.
     * @param gridHeight The height (number of rows) of the grid.
     * @param cellSize The size (width and height) of each individual cell when
     * rendering.
     * @param memoryType A string representing the memory allocation type used
     * (e.g., "Normal").
     */

    uint8_t *d_gridCurrent, *d_gridNext;
    int N = gridWidth * gridHeight;
    size_t size = N * sizeof(uint8_t);
    // * Allocate Memory on GPU

    cudaMalloc(&d_gridCurrent, size);
    cudaMalloc(&d_gridNext, size);

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

    // * Initialize the time count
    bool startCounting = false;
    int generationCount = 0;
    long long totalTime = 0;

    // * Start the simulation
    while (window.isOpen())
    {
        generationCount++;
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape))
            {
                window.close();
            }
        }

        // * Do the generation time count
        auto start = chrono::high_resolution_clock::now();

        updateGridKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_gridCurrent, d_gridNext, gridWidth, gridHeight);

        cudaDeviceSynchronize();

        auto end = chrono::high_resolution_clock::now();
        long long duration =
            chrono::duration_cast<chrono::microseconds>(end - start).count();
        totalTime += duration;

        // * END of timing count

        // * Handle the STDOUT here
        if (!startCounting && generationCount > 10)
        {
            // * We only start counting after the GPU "warms up"
            startCounting = true;
            totalTime = 0;
            generationCount = 0;
        }
        if (startCounting && generationCount == 100)
        {
            cout << "100 generations took " << totalTime
                 << " microseconds with " << threadsPerBlock
                 << " threads per block using Normal memory allocation."
                 << endl;
            generationCount = 0;
            totalTime = 0;
        }

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

void pinnedMemSimulate(RenderWindow& window, int threadsPerBlock,
                       vector<vector<uint8_t>>& gridCurrent,
                       vector<vector<uint8_t>>& gridNext, int gridWidth,
                       int gridHeight, int cellSize, string memoryType)
{
    /**
     * @brief Simulates Conway's Game of Life using pinned memory (page-locked
     * memory) on the GPU.
     *
     * This function simulates the evolution of Conway's Game of Life by
     * updating the grid based on the current state of the cells and their
     * neighbors. The simulation is run in parallel using a CUDA kernel on the
     * GPU, and the grid is allocated in pinned (page-locked) memory on the
     * host. This allows faster memory transfers between the host and the GPU.
     * The grid is rendered in a window using the SFML library.
     *
     * The function measures the processing time for every 100 generations and
     * prints the time taken to the console, excluding the time spent rendering
     * the grid.
     *
     * @param window A reference to an SFML `RenderWindow` object used for
     * rendering the grid to the screen.
     * @param threadsPerBlock The number of threads to use per CUDA block for
     * kernel execution.
     * @param gridCurrent A 2D vector of `uint8_t` representing the current
     * state of the grid, where each element is either 0 (dead) or 1 (alive).
     * @param gridNext A 2D vector of `uint8_t` where the updated state of the
     * grid will be stored.
     * @param gridWidth The width (number of columns) of the grid.
     * @param gridHeight The height (number of rows) of the grid.
     * @param cellSize The size (width and height) of each individual cell when
     * rendering.
     * @param memoryType A string representing the memory allocation type used
     * (e.g., "Pinned").
     *
     * @return void This function does not return a value. It directly modifies
     * the grid state in pinned memory on the host, performs rendering in the
     * window, and measures processing times for every 100 generations.
     */

    uint8_t *d_gridCurrent, *d_gridNext;
    int N = gridWidth * gridHeight;
    size_t size = N * sizeof(uint8_t);
    // * Allocate Memory on GPU

    cudaMallocHost(&d_gridCurrent, size);
    cudaMallocHost(&d_gridNext, size);

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

    // * Initialize the time count
    bool startCounting = false;
    int generationCount = 0;
    long long totalTime = 0;

    // * Start the simulation
    while (window.isOpen())
    {
        generationCount++;
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape))
            {
                window.close();
            }
        }
        // * Do the generation time count
        auto start = chrono::high_resolution_clock::now();

        updateGridKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_gridCurrent, d_gridNext, gridWidth, gridHeight);

        cudaDeviceSynchronize();

        auto end = chrono::high_resolution_clock::now();
        long long duration =
            chrono::duration_cast<chrono::microseconds>(end - start).count();
        totalTime += duration;

        // * END of timing count
        // * Handle the STDOUT here
        if (!startCounting && generationCount > 10)
        {
            // * We only start counting after the GPU "warms up"
            startCounting = true;
            totalTime = 0;
            generationCount = 0;
        }
        if (startCounting && generationCount == 100)
        {
            cout << "100 generations took " << totalTime
                 << " microseconds with " << threadsPerBlock
                 << " threads per block using Normal memory allocation."
                 << endl;
            generationCount = 0;
            totalTime = 0;
        }

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

void managedMemSimulate(RenderWindow& window, int threadsPerBlock,
                        vector<vector<uint8_t>>& gridCurrent,
                        vector<vector<uint8_t>>& gridNext, int gridWidth,
                        int gridHeight, int cellSize, string memoryType)
{
    /**
     * @brief Simulates Conway's Game of Life using managed memory on the GPU.
     *
     * This function runs a simulation of Conway's Game of Life on the GPU,
     * updating the grid of cells according to the game's rules. The grid is
     * allocated in managed memory, which is a special type of memory that is
     * accessible by both the CPU and the GPU. This allows for automatic
     * synchronization between the host and device memory, simplifying memory
     * management. The grid state is updated in parallel on the GPU using CUDA,
     * and the simulation is rendered using the SFML library in a window.
     *
     * The function also measures the time taken to compute 100 generations and
     * prints the processing time in microseconds to the console, excluding the
     * rendering time.
     *
     * @param window A reference to an SFML `RenderWindow` object used for
     * rendering the grid on the screen.
     * @param threadsPerBlock The number of threads to use per CUDA block for
     * kernel execution.
     * @param gridCurrent A 2D vector of `uint8_t` representing the current
     * state of the grid, where each element is either 0 (dead) or 1 (alive).
     * @param gridNext A 2D vector of `uint8_t` where the updated state of the
     * grid will be stored.
     * @param gridWidth The width (number of columns) of the grid.
     * @param gridHeight The height (number of rows) of the grid.
     * @param cellSize The size (width and height) of each individual cell when
     * rendering.
     * @param memoryType A string representing the memory allocation type used
     * (e.g., "Managed").
     *
     * @return void This function does not return a value. It directly modifies
     * the grid state in managed memory, renders the grid in the SFML window,
     * and measures processing times for every 100 generations.
     */

    uint8_t *d_gridCurrent, *d_gridNext;
    int N = gridWidth * gridHeight;
    size_t size = N * sizeof(uint8_t);
    // * Allocate Memory on GPU
    cudaMallocHost(&d_gridCurrent, size);
    cudaMallocHost(&d_gridNext, size);

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

    // * Initialize the time count
    bool startCounting = false;
    int generationCount = 0;
    long long totalTime = 0;
    // * Start the simulation
    while (window.isOpen())
    {
        generationCount++;
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape))
            {
                window.close();
            }
        }
        // * Do the generation time count
        auto start = chrono::high_resolution_clock::now();

        updateGridKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_gridCurrent, d_gridNext, gridWidth, gridHeight);

        cudaDeviceSynchronize();

        auto end = chrono::high_resolution_clock::now();
        long long duration =
            chrono::duration_cast<chrono::microseconds>(end - start).count();
        totalTime += duration;

        // * END of timing count

        // * Handle the STDOUT here
        if (!startCounting && generationCount > 10)
        {
            // * We only start counting after the GPU "warms up"
            startCounting = true;
            totalTime = 0;
            generationCount = 0;
        }
        if (startCounting && generationCount == 100)
        {
            cout << "100 generations took " << totalTime
                 << " microseconds with " << threadsPerBlock
                 << " threads per block using Normal memory allocation."
                 << endl;
            generationCount = 0;
            totalTime = 0;
        }

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
