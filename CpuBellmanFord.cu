#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "utils.h"

using namespace std;

void bellmanFord(int numVertex, int src, int* costMatrix, int* distance, int* parent) {
    distance[src] = 0;

    for (int i = 0; i < numVertex - 1; i++) {
        for (int u = 0; u < numVertex; u++) {
            for (int v = 0; v < numVertex; v++) {
                if (costMatrix[u * numVertex + v] != INF && distance[u] != INF && (distance[u] + costMatrix[u * numVertex + v]) < distance[v]) {
                    parent[v] = u;
                    distance[v] = distance[u] + costMatrix[u * numVertex + v];
                }
            }
        }
    }
}

void bellmanFord(int src, int numVertex, int* vertices, int* indices, int* edges, int* weights, int* distance, int* parent) {
    distance[src] = 0;

    for (int k = 0; k < numVertex - 1; k++) {
        cout << "k=" << k << " ";
        for (int i = 0; i < numVertex; i++) {
            // int u = vertices[i];
            for (int j = indices[i]; j < indices[i+1]; j++) {
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


int main() {

    /* Adjacency Matrix */
    
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

    //int* costMatrix = (int*) malloc(numVertex * numVertex * sizeof(int));
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

    //int* parent = (int*) malloc(numVertex * sizeof(int));
    //int* distance = (int*) malloc(numVertex * sizeof(int));

    //fill(distance, distance + numVertex, INF);
    //fill(parent, parent + numVertex, -1);

    //bellmanFord(numVertex, src, (int*)costMatrix, distance, parent);
    //printPathSSSP(numVertex, distance, parent);

    /* Adjacency Linked List */
    /*int numVertex = 264346;
    int src = 1;

    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    
    graph->numVertex = numVertex;
    graph->neighbors = (struct AdjacencyList*) malloc(numVertex * sizeof(struct AdjacencyList));
    
    for (int i = 0; i < numVertex; ++i) {
        graph->neighbors[i].head = NULL;
    }

    graph = fileToAdjacencyList(string("nyc-d.txt"), graph);


    int* costMatrix = (int*)malloc(numVertex * numVertex * sizeof(int));
    if (costMatrix == NULL) {
        cout << "malloc failed" << endl;
    }
    fill(costMatrix, costMatrix + numVertex * numVertex, INF);

    int* parent = (int*)malloc(numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * sizeof(int));
    bool* visited = (bool*)malloc(numVertex * sizeof(bool));

    fill(distance, distance + numVertex, INF);
    fill(visited, visited + numVertex, false);
    fill(parent, parent + numVertex, -1);

    dijkstra(graph, src, visited, distance, parent);
    printPathSSSP(numVertex, distance, parent);*/

    int src = 0;

    int numVertex, numEdges;
    map<int, list< pair<int, int > > > adjacencyList;
    
    fileToAdjacencyList(string("nyc-d.txt"), adjacencyList, numVertex, numEdges);

    // vector<int> vertices = { 0, 1, 2, 3, 4, 5 }, indices = { 0, 2, 5, 6, 8, 9 }, edges = { 1, 2, 2, 3, 4, 4, 4, 5, 5 }, weights = { 1,5,2,2,1,2,3,1,2 };
    vector<int> vertices, indices, edges, weights;
    /*vertices.reserve(numVertex);
    indices.reserve(numVertex + 1);
    edges.reserve(numEdges);
    weights.reserve(numEdges);*/

    adjacencyListToCSR(adjacencyList, vertices, indices, edges, weights);
    
    int* parent = (int*)malloc(numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * sizeof(int));

    fill(distance, distance + numVertex, INF);
    fill(parent, parent + numVertex, -1);
    cout << "before" << endl;

    /*for (int i = 0; i < numEdges; i++) {
        std::cout << weights[i] << ' ';
    }*/

    bellmanFord(src, numVertex, vertices.data(), indices.data(), edges.data(), weights.data(), distance, parent);
    cout << "after" << endl;
    printPathSSSP(numVertex, distance, parent);
}