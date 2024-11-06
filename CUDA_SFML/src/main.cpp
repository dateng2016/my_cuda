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

int threadsPerBlock = 32;       // -n (default to 32)
int cellSize = 5;               // -c (default to 5)
int windowWidth = 800;          // -x (default to 800)
int windowHeight = 600;         // -y (default to 600)
MemoryType memoryType = NORMAL; // -t (default to NORMAL)

int main()
{

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
