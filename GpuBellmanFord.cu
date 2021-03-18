#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaCheck.cuh"

#include <iostream>

#include "utils.h"

using namespace std;

void bellmanFord(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    distance[src] = 0;

    for (int k = 0; k < numVertex - 1; k++) {
        for (int i = 0; i < numVertex; i++) {
            // int u = vertices[i];
            for (int j = indices[i]; j < indices[i + 1]; j++) {
                int v = edges[j];
                int w = weights[j];

                ;               if (distance[i] != INF && (distance[i] + w) < distance[v]) {
                    parent[v] = i;
                    distance[v] = distance[i] + w;
                }
            }
        }
    }
}

__global__
void bellmanFordRelax(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prev_distance, int* distance, int* parent) {

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
        prev_distance[tid] = distance[tid];
    }
}

__global__
void bellmanFordParent(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
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



//__global__
//void bellmanFordUpdateDistance() {
//
//}

int main() {

    ///* Adjacency Matrix */
    //int h_costMatrix[6][6] = {
    //    {INF, 1, 5, INF, INF, INF},
    //    {INF, INF, 2, 2, 1, INF},
    //    {INF, INF, INF, INF, 2, INF},
    //    {INF, INF, INF, INF, 3, 1},
    //    {INF, INF, INF, INF, INF, 2},
    //    {INF, INF, INF, INF, INF, INF},
    //};
    //int numVertex = 6;
    //int src = 1;

    //int* costMatrix = (int*)malloc(numVertex * numVertex * sizeof(int));
    //if (costMatrix == NULL) {
    //    cout << "malloc failed" << endl;
    //}
    //fill(costMatrix, costMatrix + numVertex * numVertex, INF);

    //for (int i = 0; i < numVertex; i++) {
    //    for (int j = 0; j < numVertex; j++) {
    //        costMatrix[i * numVertex + j] = h_costMatrix[i][j];
    //    }
    //}

    //// fileToCostMatrix(string("nyc-d.txt"), numVertex, costMatrix);

    //int* parent = (int*)malloc(numVertex * sizeof(int));
    //int* distance = (int*)malloc(numVertex * sizeof(int));

    //fill(distance, distance + numVertex, INF);
    //fill(parent, parent + numVertex, -1);

    //bellmanFord(numVertex, src, (int*)costMatrix, distance, parent);
    //printPathSSSP(numVertex, distance, parent);

    // int numVertex, numEdges;
    // map<int, list< pair<int, int > > > adjacencyList;

    // fileToAdjacencyList(string("nyc-d.txt"), adjacencyList, numVertex, numEdges);
    // cout << adjacencyList.size() << " " << numVertex << " " << numEdges << endl;
    // vector<int> vertices, indices, edges, weights;
    /*vertices.reserve(numVertex);
    indices.reserve(numVertex + 1);
    edges.reserve(numEdges);
    weights.reserve(numEdges);*/
    // adjacencyListToCSR(adjacencyList, vertices, indices, edges, weights);
    /*for (auto i : weights)
        std::cout << i << ' ';*/

    // int numVertex = 6, numEdges = 9;
    // vector<int> vertices = { 0, 1, 2, 3, 4, 5 }, indices = { 0, 2, 5, 6, 8, 9 }, edges = { 1, 2, 2, 3, 4, 4, 4, 5, 5 }, weights = { 1,5,2,2,1,2,3,1,2 };
    int numVertex, numEdges;
    vector<int> vertices, indices, edges, weights;
    map<int, list< pair<int, int > > > adjacencyList;    
    fileToAdjacencyList(string("nyc-d.txt"), adjacencyList, numVertex, numEdges);
    adjacencyListToCSR(adjacencyList, vertices, indices, edges, weights);

    int src = 0;

    /*int* exp_parent = (int*)malloc(numVertex * sizeof(int));
    int* exp_distance = (int*)malloc(numVertex * sizeof(int));

    fill(exp_distance, exp_distance + numVertex, INF);
    fill(exp_parent, exp_parent + numVertex, -1);

    bellmanFord(src, numVertex, vertices.data(), indices.data(), edges.data(), weights.data(), exp_distance, exp_parent);*/

    /*int* parent = (int*)malloc(numVertex * sizeof(int));
    int* prev_distance = (int*)malloc(numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * sizeof(int));
    
    fill(prev_distance, prev_distance + numVertex, INF);
    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);


    prev_distance[src] = 0;
    distance[src] = 0;

    int* d_vertices;
    int* d_indices;
    int* d_edges;
    int* d_weights;
    int* d_prev_distance;
    int* d_distance;
    int* d_parent;
    
    cudaCheck(cudaMalloc((void**)&d_vertices, numVertex*sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_indices, (numVertex+1) * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_edges, numEdges * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_weights, numEdges * sizeof(int)));

    cudaCheck(cudaMalloc((void**)&d_prev_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_vertices, vertices.data(), numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_indices, indices.data(), (numVertex+1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_edges, edges.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weights, weights.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(d_prev_distance, prev_distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));

    for (int k = 0; k < numVertex - 1; k++) {
        bellmanFordRelax<<<(numVertex-1)/THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK>>>(numVertex, d_vertices, d_indices, d_edges, d_weights, d_prev_distance, d_distance, d_parent);
    }
    bellmanFordParent <<<(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >>> (numVertex, d_vertices, d_indices, d_edges, d_weights, d_distance, d_parent);

    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));*/

    // validateDistance(numVertex, exp_distance, distance);

    // printPathSSSP(numVertex, distance, parent);

    /*int* parent = (int*)malloc(numVertex * sizeof(int));
    int* prev_distance = (int*)malloc(numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * sizeof(int));

    fill(prev_distance, prev_distance + numVertex, INF);
    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);


    prev_distance[src] = 0;
    distance[src] = 0;

    int* d_vertices;
    int* d_indices;
    int* d_edges;
    int* d_weights;
    int* d_prev_distance;
    int* d_distance;
    int* d_parent;

    cudaCheck(cudaMalloc((void**)&d_vertices, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_indices, (numVertex + 1) * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_edges, numEdges * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_weights, numEdges * sizeof(int)));

    cudaCheck(cudaMalloc((void**)&d_prev_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_vertices, vertices.data(), numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_indices, indices.data(), (numVertex + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_edges, edges.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weights, weights.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(d_prev_distance, prev_distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));

    for (int k = 0; k < numVertex - 1; k++) {
        bellmanFordRelaxStride << <200, THREADS_PER_BLOCK >> > (numVertex, d_vertices, d_indices, d_edges, d_weights, d_prev_distance, d_distance, d_parent);
    }
    bellmanFordParentStride << <200, THREADS_PER_BLOCK >> > (numVertex, d_vertices, d_indices, d_edges, d_weights, d_distance, d_parent);

    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    printPathSSSP(numVertex, distance, parent);*/


    int* parent = (int*)malloc(numVertex * sizeof(int));
    int* prev_distance = (int*)malloc(numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * sizeof(int));
    bool* flag = (bool*)malloc(numVertex * sizeof(bool));

    fill(prev_distance, prev_distance + numVertex, INF);
    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);
    fill(flag, flag + numVertex, false);


    prev_distance[src] = 0;
    distance[src] = 0;
    flag[src] = true;

    int* d_vertices;
    int* d_indices;
    int* d_edges;
    int* d_weights;
    int* d_prev_distance;
    int* d_distance;
    int* d_parent;
    bool* d_flag;

    cudaCheck(cudaMalloc((void**)&d_vertices, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_indices, (numVertex + 1) * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_edges, numEdges * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_weights, numEdges * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_flag, numVertex * sizeof(bool)));

    cudaCheck(cudaMalloc((void**)&d_prev_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    cudaCheck(cudaMemcpy(d_vertices, vertices.data(), numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_indices, indices.data(), (numVertex + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_edges, edges.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weights, weights.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpy(d_prev_distance, prev_distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_flag, flag, numVertex * sizeof(bool), cudaMemcpyHostToDevice));

    for (int k = 0; k < numVertex - 1; k++) {
        bellmanFordRelaxStrideOptimize << <200, THREADS_PER_BLOCK >> > (numVertex, d_vertices, d_indices, d_edges, d_weights, d_prev_distance, d_distance, d_parent, d_flag);
    }
    bellmanFordParentStride << <200, THREADS_PER_BLOCK >> > (numVertex, d_vertices, d_indices, d_edges, d_weights, d_distance, d_parent);

    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    // printPathSSSP(numVertex, distance, parent);

}