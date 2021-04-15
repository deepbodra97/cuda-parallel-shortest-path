#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "cudaCheck.cuh"

#include "utils.h"

using namespace std;

/**********************************************************************************************************************************
SERIAL VERSION
***********************************************************************************************************************************/

// cpu dijkstra
void dijkstra(int src, int numVertex, int* costMatrix, int* distance, int* parent) {
    priority_queue< pair<int, int>, vector <pair<int, int>>, greater<pair<int, int>> > heap; // heap of pair<distance to node, node>
    heap.push(make_pair(0, src)); // init heap
    distance[src * numVertex + src] = 0;
    while (!heap.empty()) {
        int u = heap.top().second; // extract min
        heap.pop();

        for (int v = 0; v < numVertex ; v++) { // loop through neighbors of u
            int weight = costMatrix[u * numVertex + v]; // cost from u to v

            if (weight != INF && distance[src * numVertex + v] > distance[src * numVertex + u] + weight) { // relax
                distance[src * numVertex + v] = distance[src * numVertex + u] + weight;
                parent[src * numVertex + v] = u;
                heap.push(make_pair(distance[src * numVertex + v], v)); // add to heap
            }
        }
    }
}

// run cpu dijkstra for very source
void runCpuDijkstra(int numVertex, int* costMatrix, int* distance, int* parent) {
    // time the algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    for (int src = 0; src < numVertex; src++) { // for every source
        dijkstra(src, numVertex, costMatrix, distance, parent); // call dijkstras
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

/**********************************************************************************************************************************
NAIVE VERSION
***********************************************************************************************************************************/

// find next node to visit
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
    int src = blockIdx.x * blockDim.x + threadIdx.x; // thread src calculates shortest paths from src to every other vertex

    if (src < numVertex) {
        distance[src * numVertex + src] = 0;

        for (int i = 0; i < numVertex - 1; i++) {
            int u = extractMin(numVertex, distance, visited, src); // extract min
            if (u == -1) { // no min node to explore
                break;
            }
            visited[src * numVertex + u] = true; // mark u as visited
            for (int v = 0; v < numVertex; v++) { // loop through neighbors of u
                if (!visited[src * numVertex + v] && h_costMatrix[u * numVertex + v] != INF &&
                    distance[src * numVertex + v] > distance[src * numVertex + u] + h_costMatrix[u * numVertex + v]) { // relax

                    parent[src * numVertex + v] = u;
                    distance[src * numVertex + v] = distance[src * numVertex + u] + h_costMatrix[u * numVertex + v];
                }
            }
        }
    }
}

// run dijkstras on gpu
void runGpuDijkstra(int numVertex, int* costMatrix, bool* visited, int* distance, int* parent) {
    // time the algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    // allocate device pointers
    int* d_costMatrix;
    int* d_parent;
    int* d_distance;
    bool* d_visited;

    // allocate memory on gpu
    cudaCheck(cudaMalloc((void**)&d_costMatrix, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_visited, numVertex * numVertex * sizeof(bool)));

    // copy from cpu to gpu
    cudaCheck(cudaMemcpy(d_costMatrix, costMatrix, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_visited, visited, numVertex * numVertex * sizeof(bool), cudaMemcpyHostToDevice));

    cout << "Kernel is executing" << endl;
    dijkstraNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, d_costMatrix, d_visited, d_distance, d_parent);
    cudaCheck(cudaGetLastError()); // check if kernel launch failed
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish

    // copy from cpu to cpu
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
    string pathDataset("../data/"); // path to dataset
    string algorithm(argv[1]); // algorithm 0=cpu, 1=naive
    string pathGraphFile(pathDataset + string(argv[2])); // input file
    string validate(argv[3]); // true=compare output with cpu, false=dont
    string outputFormat(argv[4]); // none=no output (to time the kernel), print=prints path on screen, write=write output to a file in the directory named output

    int numVertex, numEdges;

    int* h_costMatrix = fileToCostMatrix(pathGraphFile, numVertex, numEdges); // convert input file to adjacency list

    int* h_parent = (int*)malloc(numVertex * numVertex * sizeof(int));
    int* h_distance = (int*)malloc(numVertex * numVertex * sizeof(int));
    bool* h_visited = (bool*)malloc(numVertex * numVertex * sizeof(bool));

    fill(h_parent, h_parent + numVertex * numVertex, -1); // fill with -1
    fill(h_distance, h_distance + numVertex * numVertex, INF); // fill with INF
    fill(h_visited, h_visited + numVertex * numVertex, false); // fill with false

    if (algorithm == "0") { // cpu version
        runCpuDijkstra(numVertex, h_costMatrix, h_distance, h_parent);
    }
    else if (algorithm == "1") { // naive
        runGpuDijkstra(numVertex, h_costMatrix, h_visited, h_distance, h_parent);
        if (validate == "true") {
            int* expParent = (int*)malloc(numVertex * numVertex * sizeof(int)); // expected parent
            int* expDistance = (int*)malloc(numVertex * numVertex * sizeof(int)); // expected distance
            fill(expDistance, expDistance + numVertex * numVertex, INF); // fill with INF
            fill(expParent, expParent + numVertex * numVertex, -1); // fill with -1
            runCpuDijkstra(numVertex, h_costMatrix, expDistance, expParent); // run on cpu
            validateDistanceAPSP(numVertex, expDistance, h_distance); // compare distance with expDistance
        }
    }

    if (outputFormat == "print") {
        printPathAPSP(numVertex, h_distance, h_parent); // print paths to screen
    }
    else if (outputFormat == "write") { // write output to a file named d{algorithm}.txt in output directory
        string pathOutputFile(string("../output/d") + algorithm + string(".txt"));
        writeOutPathAPSP(pathOutputFile, numVertex, h_distance, h_parent);
    }
    else if (outputFormat == "none") { // dont write out path

    }
    else {
        cout << "Illegal output format argument" << endl;
    }
}