#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "cudaCheck.cuh"

#include "utils.h"

using namespace std;

#define THREADS_PER_BLOCK 512

__device__
int extractMin(int numVertex, int* distance, bool* visited, int src) {
    int minNode = -1;
    int minDistance = INF;
    for (int i = 0; i < numVertex; i++) {
        if (!visited[src * numVertex + i] && distance[src * numVertex + i] < minDistance) {
            minDistance = distance[src * numVertex + i];
            minNode = i;
        }
    }
    return minNode;
}

__global__
void dijkstra(int numVertex, int* costMatrix, bool* visited, int* distance, int* parent) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src < numVertex) {
        distance[src * numVertex + src] = 0;
        parent[src * numVertex + src] = -1;

        for (int i = 0; i < numVertex - 1; i++) {
            int u = extractMin(numVertex, distance, visited, src);
            if (u == -1) { // no min node to explore
                break;
            }
            visited[src * numVertex + u] = true;
            for (int v = 0; v < numVertex; v++) {
                if (!visited[src * numVertex + v] && costMatrix[u * numVertex + v] != INF && (distance[src * numVertex + u] + costMatrix[u * numVertex + v]) < distance[src * numVertex + v]){
                    parent[src * numVertex + v] = u;
                    distance[src * numVertex + v] = distance[src * numVertex + u] + costMatrix[u * numVertex + v];
                }
            }
        }
    }
}


int main() {
    int h_numVertex = 6;
    int h_costMatrix[6][6] = {
        {INF, 1, 5, INF, INF, INF},
        {INF, INF, 2, 2, 1, INF},
        {INF, INF, INF, INF, 2, INF},
        {INF, INF, INF, INF, 3, 1},
        {INF, INF, INF, INF, INF, 2},
        {INF, INF, INF, INF, INF, INF},
    };

    int* h_parent = (int*)malloc(h_numVertex * h_numVertex * sizeof(int));
    int* h_distance = (int*)malloc(h_numVertex * h_numVertex * sizeof(int));
    bool* h_visited = (bool*)malloc(h_numVertex * h_numVertex * sizeof(bool));

    fill(h_parent, h_parent + h_numVertex * h_numVertex, -1);
    fill(h_distance, h_distance + h_numVertex * h_numVertex, INF);
    fill(h_visited, h_visited + h_numVertex * h_numVertex, false);

    const int bytesNumVertex = sizeof(int);
    const int bytesCostMatrix = h_numVertex * h_numVertex * sizeof(int);

    int* d_costMatrix;
    int* d_parent;
    int* d_distance;
    bool* d_visited;

    cudaCheck(cudaMalloc((void**)&d_costMatrix, bytesCostMatrix));
    cudaCheck(cudaMalloc((void**)&d_parent, bytesCostMatrix));
    cudaCheck(cudaMalloc((void**)&d_distance, bytesCostMatrix));
    cudaCheck(cudaMalloc((void**)&d_visited, bytesCostMatrix));

    cudaCheck(cudaMemcpy(d_costMatrix, h_costMatrix, bytesCostMatrix, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, h_parent, bytesCostMatrix, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, h_distance, bytesCostMatrix, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_visited, h_visited, bytesCostMatrix, cudaMemcpyHostToDevice));

    dijkstra<<<(h_numVertex-1)/THREADS_PER_BLOCK+1, THREADS_PER_BLOCK>>>(h_numVertex, d_costMatrix, d_visited, d_distance, d_parent);

    cudaCheck(cudaMemcpy(h_distance, d_distance, bytesCostMatrix, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_parent, d_parent, bytesCostMatrix, cudaMemcpyDeviceToHost));
    
    printPathAPSP(h_numVertex, h_distance, h_parent);
}