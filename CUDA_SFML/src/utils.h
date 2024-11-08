#pragma once
#include <iostream>
#include <vector>

using namespace std;

void seedRandomGrid(vector<vector<int>>& grid, int gridWidth, int gridHeight);
int countNeighbors(const vector<vector<int>>& grid, int x, int y, int gridWidth,
                   int gridHeight);

void parseArguments(int argc, char* argv[], int& numThreads, int& cellSize,
                    int& width, int& height, std::string& memType);
void displayConfiguration(int threadsPerBlock, int cellSize, int width,
                          int height, const std::string& memoryType);
