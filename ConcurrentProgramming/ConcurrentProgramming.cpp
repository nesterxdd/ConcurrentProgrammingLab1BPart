#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>
#include <nlohmann/json.hpp> // Make sure to include a JSON library
#include <iomanip>
#include <chrono> // Include for timing
#include <cstdlib> // For srand and rand
#include <cmath>   // For std::max
#include <algorithm> // For std::lower_bound

using json = nlohmann::json;

// Structure to hold the data entries
struct DataEntry {
    std::string ID;
    int MatrixSize;
    double Seed;
};

// Function to load data from a JSON file
void LoadData(const std::string& filePath, std::vector<DataEntry>& dataEntries) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file.");
    }

    json j;
    file >> j;

    // Populate dataEntries vector
    for (const auto& entry : j) {
        dataEntries.push_back({ entry["ID"], entry["MatrixSize"], entry["Seed"] });
    }

    file.close();
}

// Function to generate a matrix with random values
double** GenerateMatrix(int size, double seed) {
    // Allocate memory for the matrix
    double** matrix = new double* [size];
    for (int i = 0; i < size; ++i) {
        matrix[i] = new double[size];
    }

    // Seed the random number generator
    srand(static_cast<unsigned>(seed));

    // Fill the matrix with random double values between 0 and 10
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX * 10.0; // Values between 0 and 10
        }
    }
    return matrix;
}

// DFS function to find the maximum path sum from the current cell
double DFS(double** matrix, bool** visited, int x, int y, int size) {
    if (x < 0 || y < 0 || x >= size || y >= size) {
        return 0;
    }

    visited[x][y] = true;
    double maxPath = 0;

    int dx[4] = { 0, 1, 0, -1 };
    int dy[4] = { 1, 0, -1, 0 };

    for (int dir = 0; dir < 4; dir++) {
        int newX = x + dx[dir];
        int newY = y + dy[dir];

        if (newX >= 0 && newY >= 0 && newX < size && newY < size && !visited[newX][newY]) {
            maxPath = std::max(maxPath, DFS(matrix, visited, newX, newY, size));
        }
    }

    visited[x][y] = false;
    return matrix[x][y] + maxPath;
}

// Function to find the longest path sum in the given matrix
double FindLongestPathSum(double** matrix, int size) {
    double maxPathSum = 0;

    bool** visited = new bool* [size];
    for (int i = 0; i < size; ++i) {
        visited[i] = new bool[size] {false};
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            maxPathSum = std::max(maxPathSum, DFS(matrix, visited, i, j, size));
            maxPathSum = std::max(maxPathSum, DFS(matrix, visited, i, j, size));
            maxPathSum = std::max(maxPathSum, DFS(matrix, visited, i, j, size));
        }
    }

    for (int i = 0; i < size; ++i) {
        delete[] visited[i];
    }
    delete[] visited;

    return maxPathSum;
}

// Function to write results to a specified text file
void WriteResultsToFile(const std::string& filePath, const std::vector<DataEntry>& filteredEntries, double sumMatrix, double sumSeed) {
    std::ofstream outputFile(filePath);

    outputFile << "Filtered Results:\n";
    outputFile << std::left << std::setw(15) << "ID"
        << std::setw(15) << "MatrixSize"
        << std::setw(15) << "Seed" << std::endl;
    outputFile << std::string(45, '-') << std::endl;

    for (const auto& entry : filteredEntries) {
        outputFile << std::left << std::setw(15) << entry.ID
            << std::setw(15) << entry.MatrixSize
            << std::setw(15) << entry.Seed << std::endl;
    }

    outputFile << std::string(45, '-') << std::endl;
    outputFile << "Total Sum of Matrix Values: " << sumMatrix << std::endl;
    outputFile << "Total Sum of Seeds: " << sumSeed << std::endl;
    outputFile.close();
}

// Comparison function for sorting DataEntry by Seed
bool compareDataEntryBySeed(const DataEntry& a, const DataEntry& b) {
    return a.Seed < b.Seed; // Sorting by Seed
}

// Main function
int main() {
    std::vector<DataEntry> dataEntries; // Vector for data entries
    LoadData("IFU-2_NesterenkoY_L1_dat_1.json", dataEntries);

    double sumMatrixSize = 0.0;
    double sumSeed = 0.0;
    std::vector<DataEntry> filteredEntries; // Store filtered entries

    int dataCount = dataEntries.size();

    //int threadCount = std::max(2, dataCount / 4);
    int threadCount = 1;
    int chunkSize = dataCount / threadCount;
    int remainder = dataCount % threadCount;

    auto startTotal = std::chrono::high_resolution_clock::now(); // Start total timing

#pragma omp parallel num_threads(threadCount)
    {
        std::vector<DataEntry> localFilteredEntries; // Thread-local vector for filtered entries
        double localSumMatrixSize = 0.0;
        double localSumSeed = 0.0;

        int ind = omp_get_thread_num();
        int startIdx = ind * chunkSize + std::min(ind, remainder); // Adjust for remainder
        int endIdx = startIdx + chunkSize + (ind < remainder ? 1 : 0); // Adjust for last thread

        std::cout << "Thread " << ind << " processing indices " << startIdx << " to " << endIdx - 1 << std::endl;

        for (int j = startIdx; j < endIdx; ++j) {
            double** matrix = GenerateMatrix(dataEntries[j].MatrixSize, dataEntries[j].Seed);

            double longestPathSum = FindLongestPathSum(matrix, dataEntries[j].MatrixSize);

         
            if (longestPathSum > 50.0) { // Filter threshold
                localFilteredEntries.push_back(dataEntries[j]);
                localSumMatrixSize += dataEntries[j].MatrixSize;
                localSumSeed += dataEntries[j].Seed;
            }

            // Cleanup matrix memory
            for (int k = 0; k < dataEntries[j].MatrixSize; ++k) {
                delete[] matrix[k];
            }
            delete[] matrix;
        }

#pragma omp critical
        {
            for (const auto& entry : localFilteredEntries) {
                // find the position to insert to keep sorted order by Seed
                auto it = std::lower_bound(filteredEntries.begin(), filteredEntries.end(), entry, compareDataEntryBySeed);
                filteredEntries.insert(it, entry);
            }
            sumMatrixSize += localSumMatrixSize;
            sumSeed += localSumSeed;
        }
    }

    auto endTotal = std::chrono::high_resolution_clock::now(); // End total timing
    std::chrono::duration<double> elapsedTotal = endTotal - startTotal;

    // Output results to a file
    WriteResultsToFile("IFU-2_NesterenkoY_L1_rez_b.txt", filteredEntries, sumMatrixSize, sumSeed);

    std::cout << "Results written to IFU-2_NesterenkoY_L1_rez_b.txt" << std::endl;
    std::cout << "Total execution time: " << elapsedTotal.count() << " seconds" << std::endl;

    return 0;
}
