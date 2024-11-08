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

void updateGrid(vector<vector<uint8_t>>& grid, vector<vector<uint8_t>>& newGrid,
                int gridWidth, int gridHeight)
{
    /**
     * Updates the state of a 2D grid representing cells in a cellular
     * automaton, such as Conway's Game of Life. The function evaluates each
     * cell based on the current state of the grid and applies the rules of the
     * game to determine whether each cell should be alive or dead in the next
     * generation.
     *
     * @param grid A reference to a 2D vector of boolean values representing the
     *             current state of the grid. This grid remains unchanged during
     *             the function call.
     * @param newGrid A reference to a 2D vector of boolean values that will be
     *                 updated to reflect the next state of the grid based on
     * the current state and neighbor counts.
     * @param gridWidth An integer representing the width of the grid (number
     *                  of columns).
     * @param gridHeight An integer representing the height of the grid (number
     *                   of rows).
     *
     * This function does not return a value. The newGrid is modified in place
     * to represent the updated state of the grid after applying the rules of
     * the cellular automaton.
     */

    for (int x = 0; x < gridWidth; ++x)
    {
        for (int y = 0; y < gridHeight; ++y)
        {
            int neighbors = countNeighbors(grid, x, y, gridWidth, gridHeight);

            if (grid[x][y])
            {
                if (neighbors < 2 || neighbors > 3)
                {
                    newGrid[x][y] = false; // Cell dies
                }
            }
            else
            {
                if (neighbors == 3)
                {
                    newGrid[x][y] = true; // Cell becomes alive
                }
            }
        }
    }
}

void seqSimulate(RenderWindow& window, vector<vector<uint8_t>>& gridCurrent,
                 vector<vector<uint8_t>>& gridNext, int gridWidth,
                 int gridHeight)
{
    /**
     * @brief Simulates the Game of Life using a sequential algorithm.
     *
     * This function runs the simulation of Conway's Game of Life in a
     * sequential manner, updating the grid of cells based on the rules of the
     * game. It continuously processes generations of cells and displays the
     * current state in a window, while also measuring the time taken for every
     * 100 generations.
     *
     * @param window A reference to a `RenderWindow` object that handles the
     * graphical display of the simulation. This window is updated to show the
     * current state of the grid after each generation.
     * @param gridCurrent A reference to a 2D vector of boolean values
     * representing the current state of the grid, where each element indicates
     * whether a cell is alive (true) or dead (false).
     * @param gridNext A reference to a 2D vector of boolean values that will be
     * updated to reflect the next state of the grid after applying the Game of
     * Life rules.
     * @param gridWidth An integer representing the width of the grid (number of
     * columns).
     * @param gridHeight An integer representing the height of the grid (number
     * of rows).
     *
     * This function does not return a value. It modifies `gridNext` in place to
     * represent the updated state of the grid after applying the Game of Life
     * rules for each generation. Additionally, it outputs the time taken for
     * every 100 generations to the console, providing insights into the
     * performance of the simulation.
     */

    // Use this to track time.
    int generationCount = 0;
    long long totalTime = 0;
    auto start = chrono::high_resolution_clock::now();

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

        gridCurrent = gridNext;

        updateGrid(gridCurrent, gridNext, gridWidth, gridHeight);

        generationCount++;

        if (generationCount == 100)
        {
            auto end = chrono::high_resolution_clock::now();
            long long duration =
                chrono::duration_cast<chrono::microseconds>(end - start)
                    .count();
            totalTime += duration;
            cout << "100 generations took " << totalTime
                 << " microseconds with single thread." << endl;
            totalTime = 0; // Reset for next interval
            start = chrono::high_resolution_clock::now();
            generationCount = 0;
        }

        window.clear();

        for (int x = 0; x < gridWidth; ++x)
        {
            for (int y = 0; y < gridHeight; ++y)
            {
                if (gridNext[x][y])
                {
                    RectangleShape cell(Vector2f(cellSize, cellSize));
                    cell.setPosition(x * cellSize, y * cellSize);
                    cell.setFillColor(Color::White);
                    window.draw(cell);
                }
            }
        }

        window.display();
    }
}

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

    // FIXME:
    // normalMemSimulate(window, threadsPerBlock, gridCurrent, gridNext,
    // gridWidth,
    //                   gridHeight, cellSize);

    seqSimulate(window, gridCurrent, gridNext, gridWidth, gridHeight);

    return 0;
}
