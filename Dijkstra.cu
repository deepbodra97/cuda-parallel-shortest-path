#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "cudaCheck.cuh"

#include "utils.h"

using namespace std;

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
void dijkstra(int numVertex, int* h_costMatrix, bool* visited, int* distance, int* parent) {
    int src = blockIdx.x * blockDim.x + threadIdx.x;

    if (src < numVertex) {
        distance[src * numVertex + src] = 0;

        for (int i = 0; i < numVertex - 1; i++) {
            int u = extractMin(numVertex, distance, visited, src);
            if (u == -1) { // no min node to explore
                break;
            }
            visited[src * numVertex + u] = true;
            for (int v = 0; v < numVertex; v++) {
                if (!visited[src * numVertex + v] && h_costMatrix[u * numVertex + v] != INF &&
                    distance[src * numVertex + v] > distance[src * numVertex + u] + h_costMatrix[u * numVertex + v]){
                    
                    parent[src * numVertex + v] = u;
                    distance[src * numVertex + v] = distance[src * numVertex + u] + h_costMatrix[u * numVertex + v];
                }
            }
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Please provide an input file as a command line argument" << endl;
        return 0;
    }
    string pathDataset("../data/");
    string pathGraphFile(pathDataset + string(argv[1]));

    int numVertex, numEdges;

    int* h_costMatrix = fileToCostMatrix(pathGraphFile, numVertex, numEdges);

    const int bytesCostMatrix = numVertex * numVertex * sizeof(int);
    const int bytesVisited = numVertex * numVertex * sizeof(bool);

    int* h_parent = (int*)malloc(bytesCostMatrix);
    int* h_distance = (int*)malloc(bytesCostMatrix);
    bool* h_visited = (bool*)malloc(bytesVisited);

    fill(h_parent, h_parent + numVertex * numVertex, -1);
    fill(h_distance, h_distance + numVertex * numVertex, INF);
    fill(h_visited, h_visited + numVertex * numVertex, false);

    int* d_costMatrix;
    int* d_parent;
    int* d_distance;
    bool* d_visited;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    cudaCheck(cudaMalloc((void**)&d_costMatrix, bytesCostMatrix));
    cudaCheck(cudaMalloc((void**)&d_parent, bytesCostMatrix));
    cudaCheck(cudaMalloc((void**)&d_distance, bytesCostMatrix));
    cudaCheck(cudaMalloc((void**)&d_visited, bytesVisited));

    cudaCheck(cudaMemcpy(d_costMatrix, h_costMatrix, bytesCostMatrix, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, h_parent, bytesCostMatrix, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, h_distance, bytesCostMatrix, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_visited, h_visited, bytesVisited, cudaMemcpyHostToDevice));

    cout << "Kernel is executing" << endl;
    dijkstra<<<(numVertex-1)/THREADS_PER_BLOCK+1, THREADS_PER_BLOCK>>>(numVertex, d_costMatrix, d_visited, d_distance, d_parent);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(h_distance, d_distance, bytesCostMatrix, cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(h_parent, d_parent, bytesCostMatrix, cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;

    // printPathAPSP(numVertex, h_distance, h_parent);
    string pathOutputFile(string("../output/d") + string(".txt"));
    writeOutPathAPSP(pathOutputFile, numVertex, h_distance, h_parent);
}

/* test case 1
int h_costMatrix[6][6] = {
        {INF, 1, 5, INF, INF, INF},
        {INF, INF, 2, 2, 1, INF},
        {INF, INF, INF, INF, 2, INF},
        {INF, INF, INF, INF, 3, 1},
        {INF, INF, INF, INF, INF, 2},
        {INF, INF, INF, INF, INF, INF},
    };
*/