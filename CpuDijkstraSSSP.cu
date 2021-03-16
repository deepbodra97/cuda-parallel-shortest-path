#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "utils.h"

using namespace std;

int extractMin(int numVertex, int* distance, bool* visited) {
    int minNode = -1;
    int minDistance = INF;
    for (int i = 0; i < numVertex; i++) {
        if (!visited[i] && distance[i] < minDistance) {
            minDistance = distance[i];
            minNode = i;
        }
    }
    return minNode;
}

void dijkstra(int numVertex, int src, int *costMatrix, bool *visited, int * distance, int *parent) {
    distance[src] = 0;

    for (int i = 0; i < numVertex - 1; i++) {
        int u = extractMin(numVertex, distance, visited);
        if (u == -1) { // no min node to explore
            break;
        }
        visited[u] = true;
        for (int v = 0; v < numVertex; v++) {
            if (!visited[v] && costMatrix[u * numVertex + v] != INF && (distance[u] + costMatrix[u * numVertex + v]) < distance[v]){
                parent[v] = u;
                distance[v] = distance[u] + costMatrix[u * numVertex + v];
            }
        }
    }
}

void dijkstra(struct Graph* graph, int src, bool* visited, int* distance, int* parent) {
    distance[src] = 0;

    for (int i = 0; i < graph->numVertex - 1; i++) {
        int u = extractMin(graph->numVertex, distance, visited);
        if (u == -1) { // no min node to explore
            break;
        }
        visited[u] = true;
        struct AdjacencyListNode* neighbor = graph->neighbors[u].head;
        while(neighbor != NULL){
            if (!visited[neighbor->dest] && (distance[u] + neighbor->cost) < distance[neighbor->dest]) {
                parent[neighbor->dest] = u;
                distance[neighbor->dest] = distance[u] + neighbor->cost;
            }
            neighbor = neighbor->next;
        }
    }
}


//int main() {
//
//    /* Adjacency Matrix */
//    /*int numVertex = 3000;
//    int src = 1;
//
//    int* costMatrix = (int*) malloc(numVertex * numVertex * sizeof(int));
//    if (costMatrix == NULL) {
//        cout << "malloc failed" << endl;
//    }
//    fill(costMatrix, costMatrix + numVertex * numVertex, INF);
//
//    fileToCostMatrix(string("nyc-d.txt"), numVertex, costMatrix);
//
//    int* parent = (int*) malloc(numVertex * sizeof(int));
//    int* distance = (int*) malloc(numVertex * sizeof(int));
//    bool* visited = (bool*) malloc(numVertex * sizeof(bool));
//
//    fill(distance, distance + numVertex, INF);
//    fill(visited, visited + numVertex, false);
//    fill(parent, parent + numVertex, -1);
//
//    dijkstra(numVertex, src, (int*)costMatrix, visited, distance, parent);
//    printPathSSSP(numVertex, distance, parent);*/
//
//    /* Adjacency Linked List */
//    /*int numVertex = 264346;
//    int src = 1;
//
//    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
//    
//    graph->numVertex = numVertex;
//    graph->neighbors = (struct AdjacencyList*) malloc(numVertex * sizeof(struct AdjacencyList));
//    
//    for (int i = 0; i < numVertex; ++i) {
//        graph->neighbors[i].head = NULL;
//    }
//
//    graph = fileToAdjacencyList(string("nyc-d.txt"), graph);
//
//
//    int* costMatrix = (int*)malloc(numVertex * numVertex * sizeof(int));
//    if (costMatrix == NULL) {
//        cout << "malloc failed" << endl;
//    }
//    fill(costMatrix, costMatrix + numVertex * numVertex, INF);
//
//    int* parent = (int*)malloc(numVertex * sizeof(int));
//    int* distance = (int*)malloc(numVertex * sizeof(int));
//    bool* visited = (bool*)malloc(numVertex * sizeof(bool));
//
//    fill(distance, distance + numVertex, INF);
//    fill(visited, visited + numVertex, false);
//    fill(parent, parent + numVertex, -1);
//
//    dijkstra(graph, src, visited, distance, parent);
//    printPathSSSP(numVertex, distance, parent);*/
//}