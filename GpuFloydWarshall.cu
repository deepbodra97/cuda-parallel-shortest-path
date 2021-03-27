#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaCheck.cuh"

#include <iostream>

#include "utils.h"

using namespace std;

void floydWarshall(int numVertex, int* costMatrix, int* distance, int* parent) {

    for (int k = 0; k < numVertex; k++) {
        for (int i = 0; i < numVertex; i++) {
            for (int j = 0; j < numVertex; j++) {
                int itoj = i * numVertex + j;
                int itok = i * numVertex + k;
                int ktoj = k * numVertex + j;

                if (costMatrix[itok] != INF && costMatrix[ktoj] != INF && costMatrix[itoj] > costMatrix[itok] + costMatrix[ktoj]) {
                    parent[itoj] = k;
                    costMatrix[itoj] = costMatrix[itok] + costMatrix[ktoj];
                }
            }
        }
    }
}

int main() {
    int h_costMatrix[] = { 
        INF, 1, 5, INF, INF, INF,
        INF, INF, 2, 2, 1, INF,
        INF, INF, INF, INF, 2, INF,
        INF, INF, INF, INF, 3, 1,
        INF, INF, INF, INF, INF, 2,
        INF, INF, INF, INF, INF, INF,
    };

    int numVertex = 6;

    for (int i = 0; i < numVertex; i++) {
        h_costMatrix[i * numVertex + i] = 0;
    }

    int* parent = (int*)malloc(numVertex * numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * numVertex * sizeof(int));

    fill(parent, parent + numVertex * numVertex, -1);
    fill(distance, distance + numVertex * numVertex, INF);

    floydWarshall(numVertex, h_costMatrix, distance, parent);

    for (int i = 0; i < numVertex; i++) {
        for (int j = 0; j < numVertex; j++) {
            cout<<h_costMatrix[i * numVertex + j] << " ";
        }
        cout << endl;
    }
}