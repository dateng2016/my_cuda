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