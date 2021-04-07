#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaCheck.cuh"

#include <iostream>

#include "utils.h"

using namespace std;

#define TILE_DIM 32

void runCpuFloydWarshall(int numVertex, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    cout << "running the algorithm on CPU" << endl;
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

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

__global__
void floydWarshallSuperNaive(int numVertex, int k, int* distance, int* parent) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numVertex && j < numVertex) {
        int itoj = i * numVertex + j;
        int itok = i * numVertex + k;
        int ktoj = k * numVertex + j;

        if (distance[itok] != INF && distance[ktoj] != INF && distance[itoj] > distance[itok] + distance[ktoj]) {
            parent[itoj] = k;
            distance[itoj] = distance[itok] + distance[ktoj];
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

__global__
void floydWarshallTiledPhase1(int numVertex, int primary_tile_number, int* distance, int* parent) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = primary_tile_number * blockDim.y + threadIdx.y;
    int j = primary_tile_number * blockDim.x + threadIdx.x;
    if(i<numVertex && j<numVertex){
        int itoj = i * numVertex + j;
        for (int k = 0; k < TILE_DIM; k++) {
            if (i-tx+k <numVertex && j-ty+k<numVertex &&
                distance[itoj - tx + k] != INF && distance[itoj - ty * numVertex + k * numVertex] != INF &&
                distance[itoj] > distance[itoj - tx + k] + distance[itoj - ty * numVertex + k * numVertex]) {

                distance[itoj] = distance[itoj - tx + k] + distance[itoj - ty * numVertex + k * numVertex];
                parent[itoj] = TILE_DIM * primary_tile_number + k;
            }
            __syncthreads();
        }
    }
}

__global__
void floydWarshallTiledPhase2(int numVertex, int primary_tile_number, int* distance, int* parent) {
    if (blockIdx.x == primary_tile_number) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i, j;

    // 1st row of blocks for row
    if (blockIdx.y == 0) {
        i = primary_tile_number * blockDim.y + threadIdx.y;
        j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numVertex && j < numVertex) {
            int itoj = i * numVertex + j;
            for (int k = 0; k < TILE_DIM; k++) {
                if (i-tx+k-blockIdx.x * blockDim.x + primary_tile_number * blockDim.x < numVertex && j-ty+k < numVertex &&
                    distance[itoj - tx + k - blockIdx.x * blockDim.x + primary_tile_number * blockDim.x] != INF &&
                    distance[itoj - ty * numVertex + k * numVertex] != INF &&
                    distance[itoj] > distance[itoj - tx + k - blockIdx.x * blockDim.x + primary_tile_number * blockDim.x]
                    + distance[itoj - ty * numVertex + k * numVertex]) {

                    distance[itoj] = distance[itoj - tx + k - blockIdx.x * blockDim.x + primary_tile_number * blockDim.x] + distance[itoj - ty * numVertex + k * numVertex];
                    parent[itoj] = TILE_DIM * primary_tile_number + k;
                }
                __syncthreads();
            }
        }
    }

    // 2nd row of blocks for columns
    if (blockIdx.y == 1) {
        i = blockIdx.x * blockDim.y + threadIdx.y;
        j = primary_tile_number * blockDim.x + threadIdx.x;
        if (i < numVertex && j < numVertex) {
            int itoj = i * numVertex + j;
            for (int k = 0; k < TILE_DIM; k++) {
                if (i-tx+k < numVertex && j-(ty-k)- (blockIdx.x - primary_tile_number) * blockDim.x < numVertex &&
                    distance[itoj - tx + k] != INF &&
                    distance[itoj - (ty - k) * numVertex - (blockIdx.x - primary_tile_number) * blockDim.x * numVertex] != INF &&
                    distance[itoj] > distance[itoj - tx + k]
                    + distance[itoj - (ty - k) * numVertex - (blockIdx.x - primary_tile_number) * blockDim.x * numVertex]) {

                    distance[itoj] = distance[itoj - tx + k] + distance[itoj - ty * numVertex + k * numVertex - (blockIdx.x - primary_tile_number) * blockDim.x * numVertex];
                    parent[itoj] = TILE_DIM * primary_tile_number + k;
                }
                __syncthreads();
            }
        }
    }

    
}

__global__
void floydWarshallTiledPhase3(int numVertex, int primary_tile_number, int* distance, int* parent) {
    if (blockIdx.x == primary_tile_number || blockIdx.y == primary_tile_number) {
        return;
    }
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numVertex && j < numVertex) {
        int itoj = i * numVertex + j;
        for (int k = 0; k < TILE_DIM; k++) {
            if (i-tx+k - blockIdx.x * blockDim.x + primary_tile_number * blockDim.x < numVertex &&
                j-ty+k - (blockIdx.y - primary_tile_number) * blockDim.y < numVertex &&
                distance[itoj - tx + k - blockIdx.x * blockDim.x + primary_tile_number * blockDim.x] != INF &&
                distance[itoj - ty * numVertex + k * numVertex - (blockIdx.y - primary_tile_number) * blockDim.y * numVertex] != INF &&
                distance[itoj] > distance[itoj - (tx - k) - (blockIdx.x - primary_tile_number) * blockDim.x]
                + distance[itoj - (ty - k) * numVertex - (blockIdx.y - primary_tile_number) * blockDim.y * numVertex]) {

                distance[itoj] = distance[itoj - tx + k - blockIdx.x * blockDim.x + primary_tile_number * blockDim.x] + distance[itoj - ty * numVertex + k * numVertex - (blockIdx.y - primary_tile_number) * blockDim.y * numVertex];
                parent[itoj] = TILE_DIM * primary_tile_number + k;
            }
        }
    }
}

__global__ void floydWarshallTiledSharedPhase1(int numVertex, int primary_tile_number, int* distance, int* parent) {
    __shared__ int s_distance[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = TILE_DIM * primary_tile_number + ty;
    int j = TILE_DIM * primary_tile_number + tx;

    int itoj = i * numVertex + j;
   
    s_distance[ty][tx] = distance[itoj];
    
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
        __syncthreads();
        if (s_distance[ty][k] != INF &&
            s_distance[k][tx] != INF &&
            s_distance[ty][tx] > s_distance[ty][k] + s_distance[k][tx]) {

            s_distance[ty][tx] = s_distance[ty][k] + s_distance[k][tx];
        }
        __syncthreads();
    }
    distance[itoj] = s_distance[ty][tx];
}

__global__ void floydWarshallTiledSharedPhase2(int numVertex, int primary_tile_number, int* distance, int* parent) {
    if (blockIdx.x == primary_tile_number) {
        return;
    }
    __shared__ int s_distancePrimaryTile[TILE_DIM][TILE_DIM];
    __shared__ int s_distanceCurrentTile[TILE_DIM][TILE_DIM];

    int i = TILE_DIM * primary_tile_number + threadIdx.y;
    int j = TILE_DIM * primary_tile_number + threadIdx.x;
    
    int idxPrimaryTile = i * numVertex + j;
    s_distancePrimaryTile[threadIdx.y][threadIdx.x] = distance[idxPrimaryTile];
    __syncthreads();

    int idxCurrentTile;
    int shortestDistance;

    if (blockIdx.y == 0) { // 1st row of blocks for rows
        i = TILE_DIM * primary_tile_number + threadIdx.y;
        j = TILE_DIM * blockIdx.x + threadIdx.x;
        idxCurrentTile = i * numVertex + j;
        s_distanceCurrentTile[threadIdx.y][threadIdx.x] = distance[idxCurrentTile];
        __syncthreads();

        shortestDistance = s_distanceCurrentTile[threadIdx.y][threadIdx.x];

#pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            int newDistance = s_distancePrimaryTile[threadIdx.y][k] + s_distanceCurrentTile[k][threadIdx.x];
            // __syncthreads();
            if (s_distancePrimaryTile[threadIdx.y][k] != INF &&
                s_distanceCurrentTile[k][threadIdx.x] != INF &&
                newDistance < shortestDistance) {

                shortestDistance = newDistance;
            }
            __syncthreads();
        }
    } else { // 2nd row of blocks for column
        i = TILE_DIM * blockIdx.x + threadIdx.y;
        j = TILE_DIM * primary_tile_number + threadIdx.x;
        idxCurrentTile = i * numVertex + j;
        s_distanceCurrentTile[threadIdx.y][threadIdx.x] = distance[idxCurrentTile];
        __syncthreads();

        shortestDistance = s_distanceCurrentTile[threadIdx.y][threadIdx.x];

#pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            int newDistance = s_distanceCurrentTile[threadIdx.y][k] + s_distancePrimaryTile[k][threadIdx.x];
            // __syncthreads();
            if (s_distancePrimaryTile[k][threadIdx.x] != INF &&
                s_distanceCurrentTile[threadIdx.y][k] != INF &&
                newDistance < shortestDistance) {

                shortestDistance = newDistance;
            }
            __syncthreads();
        }
    }

    distance[idxCurrentTile] = shortestDistance;
}

__global__ void floydWarshallTiledSharedPhase3(int numVertex, int primary_tile_number, int* distance, int* parent) {

    if (blockIdx.x == primary_tile_number || blockIdx.y == primary_tile_number) {
        return;
    }

    __shared__ int s_distancePrimaryRow[TILE_DIM][TILE_DIM];
    __shared__ int s_distancePrimaryCol[TILE_DIM][TILE_DIM];
    __shared__ int s_distanceCurrentTile[TILE_DIM][TILE_DIM];
    
    int i, j;

    i = TILE_DIM * primary_tile_number + threadIdx.y;
    j = TILE_DIM * blockIdx.x + threadIdx.x;
    s_distancePrimaryRow[threadIdx.y][threadIdx.x] = distance[i * numVertex + j];

    i = TILE_DIM * blockIdx.y + threadIdx.y;
    j = TILE_DIM * primary_tile_number + threadIdx.x;
    s_distancePrimaryCol[threadIdx.y][threadIdx.x] = distance[i * numVertex + j];

    i = TILE_DIM * blockIdx.y + threadIdx.y;
    j = TILE_DIM * blockIdx.x + threadIdx.x;
    s_distanceCurrentTile[threadIdx.y][threadIdx.x] = distance[i * numVertex + j];

    __syncthreads();

    int shortestDist = s_distanceCurrentTile[threadIdx.y][threadIdx.x];

#pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
        int newDistance = s_distancePrimaryCol[threadIdx.y][k] + s_distancePrimaryRow[k][threadIdx.x];
        if (s_distancePrimaryCol[threadIdx.y][k] != INF &&
            s_distancePrimaryRow[k][threadIdx.x] != INF &&
            newDistance < shortestDist) {

            shortestDist = newDistance;
        }
    }
     // __syncthreads();
    distance[i * numVertex + j] = shortestDist;
}

void runFloydWarshallSuperNaive(int numVertex, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    int* d_distance;
    int* d_parent;

    // allocate memory on GPU and copy data from CPU to GPU
    cout << "allocating data on GPU" << endl;
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cout << "copying data to GPU" << endl;
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    // run kernel
    cout << "Kernel is executing" << endl;
    for (int k = 0; k < numVertex; k++) {
        floydWarshallSuperNaive << <(numVertex - 1) / TILE_DIM + 1, (numVertex - 1) / TILE_DIM + 1 >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
    }

    // copy results to CPU
    cout << "copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

void runFloydWarshallNaive(int numVertex, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    int* d_distance;
    int* d_parent;

    // allocate memory on GPU and copy data from CPU to GPU
    cout << "allocating data on GPU" << endl;   
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cout << "copying data to GPU" << endl;
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    // run kernel
    cout << "Kernel is executing" << endl;
    for (int k = 0; k < numVertex; k++) {
        floydWarshallNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
    }

    // copy results to CPU
    cout << "copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

void runFloydWarshallOptimized(int numVertex, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    int* d_distance;
    int* d_parent;

    // allocate memory on GPU and copy data from CPU to GPU
    cout << "allocating data on GPU" << endl;
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cout << "copying data to GPU" << endl;
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimGrid((numVertex - 1) / THREADS_PER_BLOCK + 1, numVertex);

    // run kernel
    cout << "Kernel is executing" << endl;
    for (int k = 0; k < numVertex; k++) {
        floydWarshallOptimized << <dimGrid, THREADS_PER_BLOCK >> > (numVertex, k, d_distance, d_parent);
    }

    // copy results to CPU
    cout << "copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

void runFloydWarshallTiled(int numVertex, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    int* d_distance;
    int* d_parent;

    // allocate memory on GPU and copy data from CPU to GPU
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cout << "copying data to GPU" << endl;
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    int numDiagonalTiles = (numVertex - 1) / TILE_DIM + 1;

    dim3 dimGridPhase1(1, 1), dimGridPhase2(numDiagonalTiles, 2), dimGridPhase3(numDiagonalTiles, numDiagonalTiles);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    cout << "Kernel is executing" << endl;
    for (int k = 0; k < numDiagonalTiles; k++) {
        cout << "Phase number " << k << endl;
        floydWarshallTiledPhase1 << <  dimGridPhase1, dimBlock >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
        floydWarshallTiledPhase2 << <  dimGridPhase2, dimBlock >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
        floydWarshallTiledPhase3 << <  dimGridPhase3, dimBlock >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
    }

    // copy results to CPU
    cout << "copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}


void runFloydWarshallTiledShared(int numVertex, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;

    cudaEventRecord(start, 0);

    int* d_distance;
    int* d_parent;

    // allocate memory on GPU and copy data from CPU to GPU
    cout << "allocating data on GPU" << endl;
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * numVertex * sizeof(int)));

    cout << "copying data to GPU" << endl;
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * numVertex * sizeof(int), cudaMemcpyHostToDevice));

    int numDiagonalTiles = (numVertex - 1) / TILE_DIM + 1;

    dim3 dimGridPhase1(1, 1), dimGridPhase2(numDiagonalTiles, 2), dimGridPhase3(numDiagonalTiles, numDiagonalTiles);
    dim3 dimBlock(TILE_DIM, TILE_DIM);

    cout << "Kernel is executing" << endl;
    for (int k = 0; k < numDiagonalTiles; k++) {
        cout << "Phase number " << k << endl;
        floydWarshallTiledSharedPhase1 << <  dimGridPhase1, dimBlock >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
        floydWarshallTiledSharedPhase2 << <  dimGridPhase2, dimBlock >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
        floydWarshallTiledSharedPhase3 << <  dimGridPhase3, dimBlock >> > (numVertex, k, d_distance, d_parent);
        cudaDeviceSynchronize();
    }

    // copy results to CPU
    cout << "copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        cout << "Please provide algorithm and input file as a command line argument" << endl;
        return 0;
    }
    string pathDataset("../data/");
    string algorithm(argv[1]);
    string pathGraphFile(pathDataset+string(argv[2]));

    int numVertex, numEdges;
    int* costMatrix = fileToCostMatrix(pathGraphFile, numVertex, numEdges);
        
    int* parent = (int*)malloc(numVertex * numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * numVertex * sizeof(int));

    APSPInitDistanceParent(numVertex, costMatrix, distance, parent);

    if (algorithm == "0") {
        runCpuFloydWarshall(numVertex, distance, parent);
    } else if (algorithm == "1") {
        runFloydWarshallSuperNaive(numVertex, distance, parent);
    } else if (algorithm == "2") {
        runFloydWarshallNaive(numVertex, distance, parent);
    } else if (algorithm == "3") {
        runFloydWarshallOptimized(numVertex, distance, parent);
    } else if (algorithm == "4") {
        runFloydWarshallTiled(numVertex, distance, parent);
    } else if (algorithm == "2") {
        runFloydWarshallTiledShared(numVertex, distance, parent);
    }
    //  printPathAPSP(numVertex, distance, parent);
    string pathOutputFile(string("../output/fw") + algorithm + string(".txt"));
    // writeOutPathAPSP(pathOutputFile, numVertex, distance, parent);
}