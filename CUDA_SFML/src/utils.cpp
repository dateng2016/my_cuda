#include <iostream>
#include <vector>
using namespace std;

void seedRandomGrid(vector<vector<bool>>& grid, int gridWidth, int gridHeight)
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
    for (int x = 0; x < gridWidth; ++x)
    {
        for (int y = 0; y < gridHeight; ++y)
        {
            grid[x][y] = (rand() % 2 == 0); // Randomly seed each pixel
        }
    }
}

int countNeighbors(const vector<vector<bool>>& grid, int x, int y,
                   int gridWidth, int gridHeight)
{
    /**
     * Counts the number of alive (true) neighbors surrounding a specified cell
     * in a 2D grid. The grid uses a toroidal wrapping behavior, meaning that
     * edges are connected to opposite edges, allowing for continuous neighbor
     * counting.
     *
     * @param grid A constant reference to a 2D vector of boolean values
     *             representing the grid, where true indicates an alive cell
     *             and false indicates a dead cell.
     * @param x The x-coordinate (column index) of the cell for which
     *          neighbors are being counted.
     * @param y The y-coordinate (row index) of the cell for which
     *          neighbors are being counted.
     * @param gridWidth An integer representing the width of the grid (number
     *                  of columns).
     * @param gridHeight An integer representing the height of the grid (number
     *                   of rows).
     *
     * @return An integer representing the total number of alive neighbors
     *         surrounding the specified cell, including wrap-around neighbors
     *         from the opposite edges of the grid.
     */

    int count = 0;
    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            if (i == 0 && j == 0)
            {
                continue;
            }
            int nx = (x + i + gridWidth) % gridWidth;
            int ny = (y + j + gridHeight) % gridHeight;
            count += grid[nx][ny];
        }
    }
    return count;
}
enum MemoryType
{
    NORMAL,
    PINNED,
    MANAGED
};
MemoryType parseMemoryType(const std::string& type)
{
    if (type == "NORMAL")
    {
        return NORMAL;
    }
    else if (type == "PINNED")
    {
        return PINNED;
    }
    else if (type == "MANAGED")
    {
        return MANAGED;
    }
    else
    {
        std::cerr << "Invalid memory type: " << type << "\n";
        exit(EXIT_FAILURE);
    }
}
