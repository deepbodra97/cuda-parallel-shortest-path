#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaCheck.cuh"

#include <iostream>

#include "utils.h"

using namespace std;

void floydWarshall(int numVertex, int* distance, int* parent) {

    for (int k = 0; k < numVertex; k++) {
        for (int i = 0; i < numVertex; i++) {
            for (int j = 0; j < numVertex; j++) {
                int itoj = i * numVertex + j;
                int itok = i * numVertex + k;
                int ktoj = k * numVertex + j;

                if (distance[itok] != INF && distance[ktoj] != INF && distance[itoj] > distance[itok] + distance[ktoj]) {
                    parent[itoj] = k;
                    distance[itoj] = distance[itok] + distance[ktoj];
                }
            }
        }
    }
}

__global__
void floydWarshallNaive() {

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

    int* parent = (int*)malloc(numVertex * numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * numVertex * sizeof(int));

    // fill(parent, parent + numVertex * numVertex, -1);
    // fill(distance, distance + numVertex * numVertex, INF);

    for (int i = 0; i < numVertex; i++) {
        for(int j = 0; j < numVertex; j++){
            if (i == j) {
                distance[i * numVertex + j] = 0;
                parent[i * numVertex + j] = -1;
            }
            else if (h_costMatrix[i * numVertex + j] == INF) {
                distance[i * numVertex + j] = INF;
                parent[i * numVertex + j] = -1;
            }
            else {
                distance[i * numVertex + j] = h_costMatrix[i * numVertex + j];
                parent[i * numVertex + j] = i;
            }
        }
    }

    floydWarshall(numVertex, distance, parent);

    for (int i = 0; i < numVertex; i++) {
        for (int j = 0; j < numVertex; j++) {
            cout<<parent[i * numVertex + j] << " ";
        }
        cout << endl;
    }
}