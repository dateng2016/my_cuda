// main.cpp
#include "cuda_kernels.cuh"
#include "utils.h"
#include <SFML/Graphics.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

using namespace std;
using namespace sf;

int threadsPerBlock = 32; // -n (default to 32)
int cellSize = 5;         // -c (default to 5)
int width = 800;          // -x (default to 800)
int height = 600;         // -y (default to 600)
string memoryType = "NORMAL";

void displayConfiguration(int threadsPerBlock, int cellSize, int width,
                          int height, const std::string& memoryType)
{
    std::cout << "Configuration:\n";
    std::cout << "  Number of threads per block: " << threadsPerBlock
              << std::endl;
    std::cout << "  Cell size: " << cellSize << std::endl;
    std::cout << "  Window width: " << width << std::endl;
    std::cout << "  Window height: " << height << std::endl;
    std::cout << "  Memory type: " << memoryType << std::endl;

    // Calculate grid size
    int gridWidth = width / cellSize;
    int gridHeight = height / cellSize;
    std::cout << "  Grid size: " << gridWidth << " x " << gridHeight
              << std::endl;
}

int main(int argc, char* argv[])
{
    // Parse command-line arguments
    parseArguments(argc, argv, threadsPerBlock, cellSize, width, height,
                   memoryType);

    // Display the configuration
    displayConfiguration(threadsPerBlock, cellSize, width, height, memoryType);
    // Set up SFML window
    sf::RenderWindow window(sf::VideoMode(800, 600), "CUDA + SFML");

    while (window.isOpen())
    {

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed ||
                Keyboard::isKeyPressed(Keyboard::Escape))
            {
                window.close();
            }
        }

        window.clear();
        // * Do the drawing here

        window.display();
    }

    return 0;
}
