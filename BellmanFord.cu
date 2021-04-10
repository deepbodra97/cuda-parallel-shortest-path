#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaCheck.cuh"

#include <iostream>

#include "utils.h"

using namespace std;

void runCpuBellmanFord(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    distance[src] = 0;

    for (int k = 0; k < numVertex - 1; k++) {
        for (int i = 0; i < numVertex; i++) {
            // int u = vertices[i];
            for (int j = indices[i]; j < indices[i + 1]; j++) {
                int v = edges[j];
                int w = weights[j];

                if (distance[i] != INF && (distance[i] + w) < distance[v]) {
                    parent[v] = i;
                    distance[v] = distance[i] + w;
                }
            }
        }
    }
}

__global__
void bellmanFordRelaxNaive(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prev_distance, int* distance, int* parent) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertex) {
        for (int j = indices[tid]; j < indices[tid + 1]; j++) {
            int v = edges[j];
            int w = weights[j];

            if (prev_distance[tid] != INF && (prev_distance[tid] + w) < prev_distance[v]) {
                // parent[v] = i; // atomic
                atomicMin(&distance[v], prev_distance[tid] + w);
            }
        }
        /*if (prev_distance[tid] > distance[tid]) {
            prev_distance[tid] = distance[tid];
        }*/
        
    }
}


__global__
void bellmanFordUpdateDistanceNaive(int numVertex, int* prev_distance, int* distance) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numVertex) {
        prev_distance[tid] = distance[tid];
        // distance[tid] = INF; // not needed technically
    }
}

__global__
void bellmanFordParentNaive(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertex) {
        for (int j = indices[tid]; j < indices[tid + 1]; j++) {
            int v = edges[j];
            int w = weights[j];

            if (distance[tid] != INF && (distance[tid] + w) == distance[v]) {
                parent[v] = tid;
            }
        }
    }
}


__global__
void bellmanFordRelaxStride(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prev_distance, int* distance, int* parent) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int idx = tid; idx < numVertex; idx += stride){
        for (int j = indices[idx]; j < indices[idx + 1]; j++) {
            int v = edges[j];
            int w = weights[j];

            if (prev_distance[idx] != INF && (prev_distance[idx] + w) < prev_distance[v]) {
                // parent[v] = i; // atomic
                atomicMin(&distance[v], prev_distance[idx] + w);
            }
        }
        /*if (prev_distance[tid] > distance[tid]) {
            prev_distance[tid] = distance[tid];
        }*/
        prev_distance[idx] = distance[idx];
    }
}

__global__
void bellmanFordUpdateDistanceStride(int numVertex, int* prev_distance, int* distance) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < numVertex; idx += stride) {
        prev_distance[tid] = distance[tid];
        // distance[tid] = INF; // not needed technically
    }
}

__global__
void bellmanFordParentStride(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < numVertex; idx += stride) {
        for (int j = indices[idx]; j < indices[idx + 1]; j++) {
            int v = edges[j];
            int w = weights[j];

            if (distance[idx] != INF && (distance[idx] + w) == distance[v]) {
                parent[v] = idx;
            }
        }
    }
}

__global__
void bellmanFordRelaxStrideOptimize(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prev_distance, int* distance, int* parent, bool* flag) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < numVertex; idx += stride) {
        if(flag[idx]){
            flag[idx] = false;
            for (int j = indices[idx]; j < indices[idx + 1]; j++) {
                int v = edges[j];
                int w = weights[j];

                if (prev_distance[idx] != INF && (prev_distance[idx] + w) < prev_distance[v]) {
                    // parent[v] = i; // atomic
                    atomicMin(&distance[v], prev_distance[idx] + w);
                }
            }
        }
        if (prev_distance[idx] > distance[idx]) {
            prev_distance[idx] = distance[idx];
            flag[idx] = true;
        }
    }
}

void runBellmanFordNaive(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    
    int* prev_distance = (int*)malloc(numVertex * sizeof(int));

    fill(prev_distance, prev_distance + numVertex, INF);
    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);

    prev_distance[src] = 0;
    distance[src] = 0;

    int* d_prev_distance;
    int* d_distance;
    int* d_parent;

    cudaCheck(cudaMalloc((void**)&d_prev_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_prev_distance, prev_distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));

    cout << "Calculating shortest distance" << endl;
    for (int k = 0; k < numVertex - 1; k++) {
        bellmanFordRelaxNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_prev_distance, d_distance, d_parent);
        bellmanFordUpdateDistanceNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, d_prev_distance, d_distance);
    }
    cout << "Constructing path" << endl;
    bellmanFordParentNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_distance, d_parent);

    cout << "Copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    
}

void runBellmanFordStride(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int* prev_distance = (int*)malloc(numVertex * sizeof(int));
    
    fill(prev_distance, prev_distance + numVertex, INF);
    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);

    prev_distance[src] = 0;
    distance[src] = 0;

    int* d_prev_distance;
    int* d_distance;
    int* d_parent;

    cudaCheck(cudaMalloc((void**)&d_prev_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_prev_distance, prev_distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));

    cout << "Calculating shortest distance" << endl;
    for (int k = 0; k < numVertex - 1; k++) {
        bellmanFordRelaxStride << <50, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_prev_distance, d_distance, d_parent);
        bellmanFordUpdateDistanceStride << <50, THREADS_PER_BLOCK >> > (numVertex, d_prev_distance, d_distance);
    }
    cout << "Constructing path" << endl;
    bellmanFordParentStride << <50, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_distance, d_parent);

    cout << "Copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
}

void runBellmanFordStrideOptimize(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {

    int* prev_distance = (int*)malloc(numVertex * sizeof(int));
    bool* flag = (bool*)malloc(numVertex * sizeof(bool));

    fill(prev_distance, prev_distance + numVertex, INF);
    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);
    fill(flag, flag + numVertex, false);


    prev_distance[src] = 0;
    distance[src] = 0;
    flag[src] = true;

    int* d_prev_distance;
    int* d_distance;
    int* d_parent;
    bool* d_flag;

    cudaCheck(cudaMalloc((void**)&d_prev_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_flag, numVertex * sizeof(bool)));

    cudaCheck(cudaMemcpy(d_prev_distance, prev_distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_flag, flag, numVertex * sizeof(bool), cudaMemcpyHostToDevice));

    cout << "Calculating shortest distance" << endl;
    for (int k = 0; k < numVertex - 1; k++) {
        bellmanFordRelaxStrideOptimize << <100, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_prev_distance, d_distance, d_parent, d_flag);
        bellmanFordUpdateDistanceStride << <100, THREADS_PER_BLOCK >> > (numVertex, d_prev_distance, d_distance);
    }
    cout << "Constructing path" << endl;
    bellmanFordParentStride << <100, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_distance, d_parent);

    cout << "Copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
}


int main(int argc, char* argv[]) {


    if (argc < 4) {
        cout << "Please provide algorithm, input file and source as a command line argument" << endl;
        return 0;
    }
    string pathDataset("../data/");
    string algorithm(argv[1]);
    string pathGraphFile(pathDataset+string(argv[2]));
    int src = stoi(argv[3]);

    int numVertex, numEdges;
    vector<int> vertices, indices, edges, weights;
    map<int, list< pair<int, int > > > adjacencyList;
    fileToAdjacencyList(pathGraphFile, adjacencyList, numVertex, numEdges);
    adjacencyListToCSR(adjacencyList, vertices, indices, edges, weights);

    int* d_vertices;
    int* d_indices;
    int* d_edges;
    int* d_weights;

    if(algorithm != "0"){
        cudaCheck(cudaMalloc((void**)&d_vertices, numVertex * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&d_indices, (numVertex + 1) * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&d_edges, numEdges * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&d_weights, numEdges * sizeof(int)));

        cudaCheck(cudaMemcpy(d_vertices, vertices.data(), numVertex * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_indices, indices.data(), (numVertex + 1) * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_edges, edges.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_weights, weights.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    int* parent = (int*)malloc(numVertex * numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * numVertex * sizeof(int));

    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);
    
    if (algorithm == "0") {
        runCpuBellmanFord(src, numVertex, vertices.data(), indices.data(), edges.data(), weights.data(), distance, parent);
    } else if (algorithm == "1") {
        runBellmanFordNaive(src, numVertex, d_vertices, d_indices, d_edges, d_weights, distance, parent);
    } else if (algorithm == "2") {
        runBellmanFordStride(src, numVertex, d_vertices, d_indices, d_edges, d_weights, distance, parent);
    } else if (algorithm == "3") {
        runBellmanFordStrideOptimize(src, numVertex, d_vertices, d_indices, d_edges, d_weights, distance, parent);
    } else {
        cout << "Illegal Algorithm" << endl;
    }

    cudaCheck(cudaFree(d_vertices));
    cudaCheck(cudaFree(d_indices));
    cudaCheck(cudaFree(d_edges));
    cudaCheck(cudaFree(d_weights));

    string pathOutputFile(string("../output/bf") + algorithm + string(".txt"));
    writeOutPathSSSP(pathOutputFile, numVertex, distance, parent);
}