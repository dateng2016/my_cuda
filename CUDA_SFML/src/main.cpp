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

int main()
{
    const int N = 1024;
    std::vector<float> A(N, 11.0f);
    std::vector<float> B(N, 2.0f);
    std::vector<float> C(N);
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << "C[" << i << "] = " << C[i] << std::endl;
    // }

    // Perform CUDA vector addition
    vectorAdd(A.data(), B.data(), C.data(), N);

    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << "C[" << i << "] = " << C[i] << std::endl;
    // }

    // Set up SFML window
    sf::RenderWindow window(sf::VideoMode(800, 600), "CUDA + SFML");

    // Visualization of result (as simple bars for each value in C)
    sf::VertexArray bars(sf::Lines, N * 2);
    float max_value = *std::max_element(C.begin(), C.end());
    for (int i = 0; i < N; ++i)
    {
        bars[i * 2].position = sf::Vector2f(i * 8, 300); // Starting point
        bars[i * 2].color = sf::Color::Red;
        bars[i * 2 + 1].position = sf::Vector2f(
            i * 8, 300 - (C[i] / max_value) * 300); // Scaled height
        bars[i * 2 + 1].color = sf::Color::Red;
    }

    std::cout << "Displaying Results..." << std::endl;
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << "C[" << i << "] = " << C[i] << std::endl;
    // }
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << "A[" << i << "] = " << A[i] << std::endl;
    // }
    // for (int i = 0; i < 10; ++i)
    // {
    //     std::cout << "B[" << i << "] = " << B[i] << std::endl;
    // }
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(bars);
        window.display();
    }

    return 0;
}
