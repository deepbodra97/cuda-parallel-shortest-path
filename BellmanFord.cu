#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaCheck.cuh"

#include <iostream>

#include "utils.h"

using namespace std;

/**********************************************************************************************************************************
SERIAL VERSION
***********************************************************************************************************************************/

// run bellman ford on cpu
void runCpuBellmanFord(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    distance[src] = 0;
    for (int k = 0; k < numVertex; k++) { // a total of numVertex iterations
        for (int i = 0; i < numVertex; i++) { // loop through all vertices
            for (int j = indices[i]; j < indices[i + 1]; j++) { // loop through neighbors of i
                int v = edges[j]; // neighbor j
                int w = weights[j]; // cost from i to j

                if (distance[i] != INF && (distance[i] + w) < distance[v]) { // relax
                    parent[v] = i;
                    distance[v] = distance[i] + w;
                }
            }
        }
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

/**********************************************************************************************************************************
NAIVE VERSION
***********************************************************************************************************************************/

// relax
__global__
void bellmanFordRelaxNaive(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prevDistance, int* distance, int* parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // thread i relaxes outgoing edges from vertex i
    if (i < numVertex) {
        for (int j = indices[i]; j < indices[i + 1]; j++) { // loop through neighbors of i
            int v = edges[j]; // neighbor j
            int w = weights[j]; // cost from i to j

            if (prevDistance[i] != INF && (prevDistance[i] + w) < distance[v]) { // relax
                atomicMin(&distance[v], prevDistance[i] + w); // atomic minimum
            }
        }
    }
}

// copy the updated cost values in prevDistance 
__global__
void bellmanFordUpdateDistanceNaive(int numVertex, int* prevDistance, int* distance) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // thread i handles vertex i
    if (i < numVertex) {
        prevDistance[i] = distance[i]; // copy distance into prevDistance
    }
}

// find parents of the vertices
__global__
void bellmanFordParentNaive(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // thread i checks if it is the parent of any of it's neighbors
    if (i < numVertex) {
        for (int j = indices[i]; j < indices[i + 1]; j++) { // loop through neighbors of i
            int v = edges[j]; // neighbor j
            int w = weights[j]; // cost from i to j

            if (distance[i] != INF && (distance[i] + w) == distance[v]) {
                parent[v] = i;
            }
        }
    }
}

// run naive version of bellmand ford
void runBellmanFordNaive(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int* prevDistance = (int*)malloc(numVertex * sizeof(int));

    fill(prevDistance, prevDistance + numVertex, INF); // fill with INF

    prevDistance[src] = 0;
    distance[src] = 0;

    // time the algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    // device pointers
    int* d_prevDistance;
    int* d_distance;
    int* d_parent;

    // allocate memory on device
    cudaCheck(cudaMalloc((void**)&d_prevDistance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    // copy from cpu to gpus
    cudaCheck(cudaMemcpy(d_prevDistance, prevDistance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));

    cout << "Calculating shortest distance" << endl;
    for (int k = 0; k < numVertex - 1; k++) { // numVertex-1 iterations
        bellmanFordRelaxNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_prevDistance, d_distance, d_parent);
        cudaCheck(cudaGetLastError()); // check if kernel launch failed
        cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
        bellmanFordUpdateDistanceNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, d_prevDistance, d_distance);
        cudaCheck(cudaGetLastError()); // check if kernel launch failed
        cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finis
    }
    cout << "Constructing path" << endl;
    bellmanFordParentNaive << <(numVertex - 1) / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_distance, d_parent);
    cudaCheck(cudaGetLastError()); // check if kernel launch failed
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finis

    // copy from gpu to cpu
    cout << "Copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_prevDistance));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}

/**********************************************************************************************************************************
STRIDE VERSION
***********************************************************************************************************************************/

// relax with stride
__global__
void bellmanFordRelaxStride(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prevDistance, int* distance, int* parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread tid relaxes outgoing edges from vertex tid, tid+stride, tid+2*stride,...
    int stride = blockDim.x * gridDim.x; // stride length

    for(int i = tid; i < numVertex; i += stride){
        for (int j = indices[i]; j < indices[i + 1]; j++) {
            int v = edges[j]; // neighbor j
            int w = weights[j]; // cost from i to j

            if (prevDistance[i] != INF && (prevDistance[i] + w) < distance[v]) { //relax
                atomicMin(&distance[v], prevDistance[i] + w);
            }
        }
    }
}

// copy the updated cost values in prevDistance with stride
__global__
void bellmanFordUpdateDistanceStride(int numVertex, int* prevDistance, int* distance) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread tid handles vertex tid, tid+stride, tid+2*stride, ...
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numVertex; i += stride) {
        prevDistance[i] = distance[i]; // copy distance into prevDistance
    }
}

// find parents of the vertices with stride
__global__
void bellmanFordParentStride(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numVertex; i += stride) {
        for (int j = indices[i]; j < indices[i + 1]; j++) { // loop through neighbors of i
            int v = edges[j]; // neighbor j
            int w = weights[j]; // cost from i to j

            if (distance[i] != INF && (distance[i] + w) == distance[v]) {
                parent[v] = i;
            }
        }
    }
}

// run stride version of bellmand ford
void runBellmanFordStride(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int* prevDistance = (int*)malloc(numVertex * sizeof(int));

    fill(prevDistance, prevDistance + numVertex, INF); // fill with INF

    prevDistance[src] = 0;
    distance[src] = 0;

    // time the algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    // device pointers
    int* d_prevDistance;
    int* d_distance;
    int* d_parent;


    // allocate memory on device
    cudaCheck(cudaMalloc((void**)&d_prevDistance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));

    // copy from cpu to gpu
    cudaCheck(cudaMemcpy(d_prevDistance, prevDistance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));

    int numBlocks = ((numVertex - 1) / THREADS_PER_BLOCK + 1) / 2; // use half the number of required blocks
    cout << "Calculating shortest distance" << endl;
    for (int k = 0; k < numVertex - 1; k++) { // numVertex-1 iterations
        bellmanFordRelaxStride << <numBlocks, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_prevDistance, d_distance, d_parent);
        cudaCheck(cudaGetLastError()); // check if kernel launch failed
        cudaCheck(cudaDeviceSynchronize()); // wait for the kernel to finish
        bellmanFordUpdateDistanceStride << <numBlocks, THREADS_PER_BLOCK >> > (numVertex, d_prevDistance, d_distance);
        cudaCheck(cudaGetLastError()); // check if kernel launch failed
        cudaCheck(cudaDeviceSynchronize()); // wait for the kernel to finish
    }
    cout << "Constructing path" << endl;
    bellmanFordParentStride << <numBlocks, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_distance, d_parent);
    cudaCheck(cudaGetLastError()); // check if kernel launch failed
    cudaCheck(cudaDeviceSynchronize()); // wait for the kernel to finish

    // copy from gpu to cpu
    cout << "Copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_prevDistance));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}



/**********************************************************************************************************************************
STRIDE WITH FLAG VERSION
***********************************************************************************************************************************/

// relax with stride
__global__
void bellmanFordRelaxStrideFlag(int numVertex, int* vertices, int* indices, int* edges, int* weights, int* prevDistance, int* distance, int* parent, bool* flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread i relaxes outgoing edges from vertex i
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numVertex; i += stride) {
        if (flag[i]) { // relax outgoing edges of i only if distance to i changed in the previous iteration
            flag[i] = false;
            for (int j = indices[i]; j < indices[i + 1]; j++) { // loop through neighbors of i
                int v = edges[j]; // neighbor j
                int w = weights[j]; // cost from i to j

                if (prevDistance[i] != INF && (prevDistance[i] + w) < distance[v]) { // relax
                    atomicMin(&distance[v], prevDistance[i] + w);
                }
            }
        }
    }
}

// copy the updated cost values in prevDistance with stride and set flag to true if the cost to i was changed in the current iteration
__global__
void bellmanFordUpdateDistanceStrideFlag(int numVertex, int* prevDistance, int* distance, bool* flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numVertex; i += stride) {
        if (prevDistance[i] > distance[i]) {
            flag[i] = true;
        }
        prevDistance[i] = distance[i];
    }
}

// run stride with flag version of bellmand ford
void runBellmanFordStrideFlag(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    int* prevDistance = (int*)malloc(numVertex * sizeof(int));
    bool* flag = (bool*)malloc(numVertex * sizeof(bool));

    fill(prevDistance, prevDistance + numVertex, INF); // fill with INF
    fill(flag, flag + numVertex, false); // fill with false

    prevDistance[src] = 0;
    distance[src] = 0;
    flag[src] = true;

    // time the algorithm
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float duration;
    cudaEventRecord(start, 0);

    // device pointers
    int* d_prevDistance;
    int* d_distance;
    int* d_parent;
    bool* d_flag;

    // allocate memory on gpu
    cudaCheck(cudaMalloc((void**)&d_prevDistance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_distance, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_parent, numVertex * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&d_flag, numVertex * sizeof(bool)));

    // copy from cpu to cpu
    cudaCheck(cudaMemcpy(d_prevDistance, prevDistance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_distance, distance, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parent, parent, numVertex * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_flag, flag, numVertex * sizeof(bool), cudaMemcpyHostToDevice));

    cout << "Calculating shortest distance" << endl;
    int numBlocks = ((numVertex - 1) / THREADS_PER_BLOCK + 1) / 2; // use half the number of required blocks
    for (int k = 0; k < numVertex - 1; k++) { // numVertex-1 iterations
        bellmanFordRelaxStrideFlag << <numBlocks, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_prevDistance, d_distance, d_parent, d_flag);
        cudaCheck(cudaGetLastError()); // check if kernel launch failed
        cudaCheck(cudaDeviceSynchronize()); // wait for the kernel to finish
        bellmanFordUpdateDistanceStrideFlag << <numBlocks, THREADS_PER_BLOCK >> > (numVertex, d_prevDistance, d_distance, d_flag);
        cudaCheck(cudaGetLastError()); // check if kernel launch failed
        cudaCheck(cudaDeviceSynchronize()); // wait for the kernel to finish
    }
    cout << "Constructing path" << endl;
    bellmanFordParentStride << <numBlocks, THREADS_PER_BLOCK >> > (numVertex, vertices, indices, edges, weights, d_distance, d_parent);
    cudaCheck(cudaGetLastError()); // check if kernel launch failed
    cudaCheck(cudaDeviceSynchronize()); // wait for the kernel to finish

    // copy from gpu to cpu
    cout << "Copying results to CPU" << endl;
    cudaCheck(cudaMemcpy(distance, d_distance, numVertex * sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(parent, d_parent, numVertex * sizeof(int), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_prevDistance));
    cudaCheck(cudaFree(d_flag));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);
    cout << "Time: " << duration << "ms" << endl;
}


int main(int argc, char* argv[]) {

    if (argc < 6) {
        cout << "Please provide algorithm, input file, source and validate in the command line argument" << endl;
        return 0;
    }
    string pathDataset("../data/"); // path to dataset
    string algorithm(argv[1]); // algorithm 0=cpu, 1=naive, 2=stride, 3=stride with flag
    string pathGraphFile(pathDataset+string(argv[2])); // input file
    int src = stoi(argv[3]); // source node in the range [0, n-1]
    string validate(argv[4]); // true=compare output with cpu, false=dont
    string outputFormat(argv[4]); // none=no output (to time the kernel), print=prints path on screen, write=write output to a file in the directory named output

    int numVertex, numEdges;
    vector<int> vertices, indices, edges, weights; // for CSR format of a graph
    map<int, list< pair<int, int > > > adjacencyList; // adjaceny list of a graph
    fileToAdjacencyList(pathGraphFile, adjacencyList, numVertex, numEdges); // convert input file to adjacency list
    adjacencyListToCSR(adjacencyList, vertices, indices, edges, weights); // convert adjacency list to CSR format

    adjacencyList.clear(); // clear adjacency list

    int* d_vertices;
    int* d_indices;
    int* d_edges;
    int* d_weights;

    if(algorithm != "0"){ // copy data to gpu if needed
        cudaCheck(cudaMalloc((void**)&d_vertices, numVertex * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&d_indices, (numVertex + 1) * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&d_edges, numEdges * sizeof(int)));
        cudaCheck(cudaMalloc((void**)&d_weights, numEdges * sizeof(int)));

        cudaCheck(cudaMemcpy(d_vertices, vertices.data(), numVertex * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_indices, indices.data(), (numVertex + 1) * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_edges, edges.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_weights, weights.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    int* parent = (int*)malloc(numVertex * sizeof(int)); // parent of a vertex for path finding
    int* distance = (int*)malloc(numVertex * sizeof(int)); // distance from src to a vertex
    
    fill(distance, distance + numVertex, INF); // fill with INF
    fill(parent, parent + numVertex, -1); // fill with -1

    if (algorithm == "0") { // cpu version
        runCpuBellmanFord(src, numVertex, vertices.data(), indices.data(), edges.data(), weights.data(), distance, parent);
    } else{
        if (algorithm == "1") { // naive
            runBellmanFordNaive(src, numVertex, d_vertices, d_indices, d_edges, d_weights, distance, parent);
        }
        else if (algorithm == "2") { // stride
            runBellmanFordStride(src, numVertex, d_vertices, d_indices, d_edges, d_weights, distance, parent);
        }
        else if (algorithm == "3") { // stride with flag
            runBellmanFordStrideFlag(src, numVertex, d_vertices, d_indices, d_edges, d_weights, distance, parent);
        }
        else {
            cout << "Illegal Algorithm" << endl;
        }

        if (validate == "true") { // validate gpu output with cpu
            int* expParent = (int*)malloc(numVertex * sizeof(int)); // expected parent
            int* expDistance = (int*)malloc(numVertex * sizeof(int)); // expected distance
            fill(expDistance, expDistance + numVertex, INF); // fill with INF
            fill(expParent, expParent + numVertex, -1); // fill with -1
            runCpuBellmanFord(src, numVertex, vertices.data(), indices.data(), edges.data(), weights.data(), distance, expParent); // run on cpu
            validateDistanceSSSP(numVertex, expDistance, distance); // compare distance with expDistance
        }
    }
        
    // free
    cudaCheck(cudaFree(d_vertices));
    cudaCheck(cudaFree(d_indices));
    cudaCheck(cudaFree(d_edges));
    cudaCheck(cudaFree(d_weights));

    if (outputFormat == "print") {
        printPathSSSP(numVertex, distance, parent); // print paths to screen
    }
    else if (outputFormat == "write") { // write output to a file named bf{algorithm}.txt in output directory
        string pathOutputFile(string("../output/bf") + algorithm + string(".txt"));
        writeOutPathSSSP(pathOutputFile, numVertex, distance, parent);
    }
    else if (outputFormat == "none") { // dont write out path
        
    }
    else {
        cout << "Illegal output format argument" << endl;
    }
}