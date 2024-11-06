#include <iostream>
using namespace std;
#include <vector>

void seedRandomGrid(vector<vector<bool>>& grid, int gridWidth, int gridHeight);
int countNeighbors(const vector<vector<bool>>& grid, int x, int y,
                   int gridWidth, int gridHeight);

void parseArguments(int argc, char* argv[], int& numThreads, int& cellSize,
                    int& width, int& height, std::string& memType);