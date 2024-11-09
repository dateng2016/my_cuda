/*
Author: Da Teng
Class: ECE6122
Last Date Modified: date 11/8/2024
Description:
This is header file for the utility function
*/
#pragma once
#include <iostream>
#include <vector>

using namespace std;

void seedRandomGrid(vector<vector<uint8_t>>& grid, int gridWidth,
                    int gridHeight);
int countNeighbors(const vector<vector<uint8_t>>& grid, int x, int y,
                   int gridWidth, int gridHeight);

void parseArguments(int argc, char* argv[], int& numThreads, int& cellSize,
                    int& width, int& height, std::string& memType);
void displayConfiguration(int threadsPerBlock, int cellSize, int width,
                          int height, const std::string& memoryType);
