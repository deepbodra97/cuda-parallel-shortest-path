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
            if (!visited[v] && costMatrix[u * numVertex + v] != INF && (distance[u] + costMatrix[u * numVertex + v]) < distance[v])
            {
                parent[v] = u;
                distance[v] = distance[u] + costMatrix[u * numVertex + v];
            }
        }
    }
}


int main() {
    int numVertex = 6;
    int costMatrix[6][6] = {
        {INF, 1, 5, INF, INF, INF},
        {INF, INF, 2, 2, 1, INF},
        {INF, INF, INF, INF, 2, INF},
        {INF, INF, INF, INF, 3, 1},
        {INF, INF, INF, INF, INF, 2},
        {INF, INF, INF, INF, INF, INF},
    };
    int src = 1;

    int* parent = (int*)malloc(numVertex * sizeof(int));
    int* distance = (int*)malloc(numVertex * sizeof(int));
    bool* visited = (bool*)malloc(numVertex * sizeof(bool));

    fill(distance, distance + numVertex, INF);
    fill(visited, visited + numVertex, false);
    fill(parent, parent + numVertex, -1);

    dijkstra(numVertex, src, (int*)costMatrix, visited, distance, parent);
    printPathSSSP(numVertex, distance, parent);
}