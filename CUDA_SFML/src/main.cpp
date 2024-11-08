// main.cpp
#include "cuda_kernels.cuh"
#include "utils.h"
#include <SFML/Graphics.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace sf;

int threadsPerBlock = 32; // -n (default to 32)
int cellSize = 5;         // -c (default to 5)
int windowWidth = 800;    // -x (default to 800)
int windowHeight = 600;   // -y (default to 600)
string memoryType = "NORMAL";

int main(int argc, char* argv[])
{
    // Parse command-line arguments
    parseArguments(argc, argv, threadsPerBlock, cellSize, windowWidth,
                   windowHeight, memoryType);

    // Display the configuration
    displayConfiguration(threadsPerBlock, cellSize, windowWidth, windowHeight,
                         memoryType);

    // * Calculate Grid Width and Height
    int gridWidth = windowWidth / cellSize;
    int gridHeight = windowHeight / cellSize;

    // * Set up SFML window
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight),
                            "CUDA + SFML");
    window.setFramerateLimit(120); // Set frame rate to control speed
    vector<vector<uint8_t>> gridCurrent(gridHeight,
                                        vector<uint8_t>(gridWidth, false));
    vector<vector<uint8_t>> gridNext(gridHeight,
                                     vector<uint8_t>(gridWidth, false));

    seedRandomGrid(gridCurrent, gridWidth, gridHeight);
    gridNext = gridCurrent;

    if (memoryType == "NORMAL")
    {
        normalMemSimulate(window, threadsPerBlock, gridCurrent, gridNext,
                          gridWidth, gridHeight, cellSize, memoryType);
    }
    else if (memoryType == "PINNED")
    {
        cout << "hello" << endl;
        pinnedMemSimulate(window, threadsPerBlock, gridCurrent, gridNext,
                          gridWidth, gridHeight, cellSize, memoryType);
    }
    else if (memoryType == "MANAGED")
    {
        managedMemSimulate(window, threadsPerBlock, gridCurrent, gridNext,
                           gridWidth, gridHeight, cellSize, memoryType);
    }

    return 0;
}
