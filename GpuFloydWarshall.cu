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
void floydWarshallNaive(int numVertex, int k, int* distance, int* parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numVertex) {
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

__global__
void floydWarshallOptimized(int numVertex, int k, int* distance, int* parent) {//G will be the adjacency matrix, P will be path matrix
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < numVertex){
        int itoj = numVertex * i + j;
        int itok = numVertex * i + k;
        int ktoj = numVertex * k + j;

        __shared__ int dist_itok;
        if (threadIdx.x == 0){
            dist_itok = distance[itok];
        }
        __syncthreads();

        if (dist_itok != INF && distance[ktoj] != INF && distance[itoj] > dist_itok + distance[ktoj]) {
            distance[itoj] = dist_itok + distance[ktoj];
            parent[itoj] = k;
        }
    }
}

void runFloydWarshallNaive(int numVertex, int* distance, int* parent) {
    int* d_distance;
    int* d_parent;

    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    for (int k = 0; k < numVertex; k++) {
        floydWarshallNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, k, d_distance, d_parent);
    }

    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
}

void runFloydWarshallOptimized(int numVertex, int* distance, int* parent) {
    int* d_distance;
    int* d_parent;

    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimGrid((numVertex - 1) / THREADS_PER_BLOCK + 1, numVertex);

    for (int k = 0; k < numVertex; k++) {
        floydWarshallOptimized << <dimGrid, THREADS_PER_BLOCK >> > (numVertex, k, d_distance, d_parent);
    }

    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
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

    // floydWarshall(numVertex, distance, parent);

    // runFloydWarshallNaive(numVertex, distance, parent);
    
    runFloydWarshallOptimized(numVertex, distance, parent);

    for (int i = 0; i < numVertex; i++) {
        for (int j = 0; j < numVertex; j++) {
            cout<<distance[i * numVertex + j] << " ";
        }
        cout << endl;
    }
}