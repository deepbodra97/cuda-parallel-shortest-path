#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "cudaCheck.cuh"

#include "utils.h"

using namespace std;

void dijkstra(int src, int numVertex, int* costMatrix, int* distance, int* parent) {
    priority_queue< pair<int, int>, vector <pair<int, int>>, greater<pair<int, int>> > heap;

    heap.push(make_pair(0, src));
    distance[src * numVertex + src] = 0;
    while (!heap.empty()) {
        int u = heap.top().second;
        heap.pop();

        for (int v = 0; v < numVertex ; v++) {
            int weight = costMatrix[u * numVertex + v];

            if (weight != INF && distance[src * numVertex + v] > distance[src * numVertex + u] + weight) {
                distance[src * numVertex + v] = distance[src * numVertex + u] + weight;
                parent[src * numVertex + v] = u;
                heap.push(make_pair(distance[src * numVertex + v], v));
            }
        }
    }
}

void runCpuDijkstra(int numVertex, int* costMatrix, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    for (int src = 0; src < numVertex; src++) {
        dijkstra(src, numVertex, costMatrix, distance, parent);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

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
void dijkstraNaive(int numVertex, int* h_costMatrix, bool* visited, int* distance, int* parent) {
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
                    distance[src * numVertex + v] > distance[src * numVertex + u] + h_costMatrix[u * numVertex + v]) {

                    parent[src * numVertex + v] = u;
                    distance[src * numVertex + v] = distance[src * numVertex + u] + h_costMatrix[u * numVertex + v];
                }
            }
        }
    }
}


void runGpuDijkstra(int numVertex, int* costMatrix, bool* visited, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;


    cudaEventRecord(start, 0);

    int* d_costMatrix;
    int* d_parent;
    int* d_distance;
    bool* d_visited;

    cudaCheck(cudaMalloc((void**)&d_costMatrix, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_visited, numVertex * numVertex * sizeof(bool)));

    cudaCheck(cudaMemcpy(d_costMatrix, costMatrix, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_visited, visited, numVertex * numVertex * sizeof(bool), cudaMemcpyHostToDevice));

    cout << "Kernel is executing" << endl;
    dijkstraNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, d_costMatrix, d_visited, d_distance, d_parent);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cout << "Please provide an input file as a command line argument" << endl;
        return 0;
    }
    string pathDataset("../data/");
    string algorithm(argv[1]);
    string pathGraphFile(pathDataset + string(argv[2]));
    string validate(argv[3]);
    string outputFormat(argv[4]);

    int numVertex, numEdges;

    int* h_costMatrix = fileToCostMatrix(pathGraphFile, numVertex, numEdges);

    int* h_parent = (int*)malloc(numVertex * numVertex * sizeof(int));
    int* h_distance = (int*)malloc(numVertex * numVertex * sizeof(int));
    bool* h_visited = (bool*)malloc(numVertex * numVertex * sizeof(bool));

    fill(h_parent, h_parent + numVertex * numVertex, -1);
    fill(h_distance, h_distance + numVertex * numVertex, INF);
    fill(h_visited, h_visited + numVertex * numVertex, false);

    

    if (algorithm == "0") {
        runCpuDijkstra(numVertex, h_costMatrix, h_distance, h_parent);
    }
    else if (algorithm == "1") {
        runGpuDijkstra(numVertex, h_costMatrix, h_visited, h_distance, h_parent);
        if (validate == "true") {
            int* exp_parent = (int*)malloc(numVertex * numVertex * sizeof(int));
            int* exp_distance = (int*)malloc(numVertex * numVertex * sizeof(int));
            fill(exp_distance, exp_distance + numVertex * numVertex, INF);
            fill(exp_parent, exp_parent + numVertex * numVertex, -1);
            runCpuDijkstra(numVertex, h_costMatrix, h_distance, h_parent);
            validateDistanceAPSP(numVertex, exp_distance, h_distance);
        }
    }

    // printPathAPSP(numVertex, h_distance, h_parent);
    

    if (outputFormat == "print") {
        printPathAPSP(numVertex, h_distance, h_parent);
    }
    else if (outputFormat == "write") {
        string pathOutputFile(string("../output/d") + algorithm + string(".txt"));
        writeOutPathAPSP(pathOutputFile, numVertex, h_distance, h_parent);
    }
    else if (outputFormat == "none") {

    }
    else {
        cout << "Illegal output format argument" << endl;
    }
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