/*
Author: Da Teng
Class: ECE6122
Last Date Modified: date 11/8/2024
Description:
This file is used for storing some utility function needed for the lab
*/
#include <iostream>
#include <vector>
using namespace std;
#include <cassert>
#include <cstring>

void seedRandomGrid(vector<vector<uint8_t>>& grid, int gridWidth,
                    int gridHeight)
{
    /**
     * Seeds a 2D grid with random boolean values, representing cells in a
     * cellular automaton. Each cell is randomly initialized to either alive
     * (true) or dead (false) state.
     *
     * @param grid A reference to a 2D vector of boolean values that will be
     *             modified in place. The size of the grid should match the
     *             specified gridWidth and gridHeight parameters.
     * @param gridWidth An integer representing the width of the grid (number
     *                  of columns).
     * @param gridHeight An integer representing the height of the grid (number
     *                   of rows).
     *
     * This function does not return a value. The grid is modified directly
     * as it is seeded with random values.
     */

    srand(static_cast<unsigned>(time(nullptr)));
    for (int y = 0; y < gridHeight; ++y)
    {
        for (int x = 0; x < gridWidth; ++x)
        {
            grid[y][x] = (rand() % 2 == 0); // Randomly seed each pixel
        }
    }
}

void parseArguments(int argc, char* argv[], int& threadsPerBlock, int& cellSize,
                    int& windowWidth, int& windowHeight,
                    std::string& memoryType)
{
    /**
     * @brief Parses command-line arguments and sets simulation configuration.
     *
     * This function processes command-line arguments passed to the program and
     * uses them to configure simulation parameters such as the number of
     * threads per block, cell size, window dimensions, and memory allocation
     * type. The function verifies that the arguments are valid and ensures that
     * certain parameters meet specific constraints (e.g., number of threads
     * must be a multiple of 32).
     *
     * @param argc The number of command-line arguments passed to the program.
     * @param argv The array of command-line argument strings.
     * @param threadsPerBlock An integer reference that will be set to the
     * number of threads per block.
     * @param cellSize An integer reference that will be set to the size of each
     * cell in the simulation.
     * @param windowWidth An integer reference that will be set to the width of
     * the simulation window.
     * @param windowHeight An integer reference that will be set to the height
     * of the simulation window.
     * @param memoryType A string reference that will be set to the memory type
     * (e.g., "NORMAL", "PINNED", "MANAGED").
     *
     * @return void This function does not return any value. It modifies the
     * input parameters by reference.
     *
     * @throws std::invalid_argument If an invalid memory type is provided
     * (other than "NORMAL", "PINNED", or "MANAGED").
     * @throws std::logic_error If any argument fails the assertion checks
     * (e.g., threadsPerBlock is not a multiple of 32).
     */

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (std::strcmp(argv[i], "-n") == 0)
        {
            if (i + 1 < argc)
            {
                threadsPerBlock = std::stoi(argv[++i]);
                assert(threadsPerBlock % 32 == 0 &&
                       "Number of threads must be a multiple of 32.");
            }
        }
        else if (std::strcmp(argv[i], "-c") == 0)
        {
            if (i + 1 < argc)
            {
                cellSize = std::stoi(argv[++i]);
                assert(cellSize >= 1 && "Cell size must be >= 1.");
            }
        }
        else if (std::strcmp(argv[i], "-x") == 0)
        {
            if (i + 1 < argc)
            {
                windowWidth = std::stoi(argv[++i]);
            }
        }
        else if (std::strcmp(argv[i], "-y") == 0)
        {
            if (i + 1 < argc)
            {
                windowHeight = std::stoi(argv[++i]);
            }
        }
        else if (std::strcmp(argv[i], "-t") == 0)
        {
            if (i + 1 < argc)
            {
                memoryType = argv[++i];
                // Ensure memory type is valid
                if (memoryType != "NORMAL" && memoryType != "PINNED" &&
                    memoryType != "MANAGED")
                {
                    std::cerr << "Invalid memory type. Valid options are: "
                                 "NORMAL, PINNED, or MANAGED."
                              << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
    }
}

void displayConfiguration(int threadsPerBlock, int cellSize, int windowWidth,
                          int windowHeight, const std::string& memoryType)
{
    /**
     * @brief Displays the current configuration of the simulation.
     *
     * This function prints out the configuration settings for the Game of Life
     * simulation, including the number of threads per block, the size of each
     * cell, the window dimensions, and the type of memory used for GPU
     * computations. It also calculates and displays the grid size based on the
     * window dimensions and cell size.
     *
     * @param threadsPerBlock The number of threads to be used per block in the
     * CUDA kernel.
     * @param cellSize The size of each individual cell, in pixels, used for
     * rendering.
     * @param windowWidth The width of the window in pixels.
     * @param windowHeight The height of the window in pixels.
     * @param memoryType A string representing the type of memory allocation
     * used (e.g., "Normal", "Pinned", "Managed").
     *
     * @return void This function does not return any value. It only prints the
     * configuration information to the console.
     */

    std::cout << "Configuration:\n";
    std::cout << "  Number of threads per block: " << threadsPerBlock
              << std::endl;
    std::cout << "  Cell size: " << cellSize << std::endl;
    std::cout << "  Window width: " << windowWidth << std::endl;
    std::cout << "  Window height: " << windowHeight << std::endl;
    std::cout << "  Memory type: " << memoryType << std::endl;

    // Calculate grid size
    int gridWidth = windowWidth / cellSize;
    int gridHeight = windowHeight / cellSize;
    std::cout << "  Grid size: " << gridWidth << " x " << gridHeight
              << std::endl;
}