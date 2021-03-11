#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

#define INF INT_MAX

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

void printPath(int numVertex, int* distance, int* parent) {
    cout << "Node\tCost\tPath"<<endl;
    for (int i = 0; i < numVertex; i++) {
        if(distance[i] != INF){
            cout << i << "\t" << distance[i] << "\t";
            cout << i;

            int tmp = parent[i];
            while (tmp != -1)
            {
                cout << "<-" << tmp;
                tmp = parent[tmp];
            }
        }
        else {
            cout << i << "\t" << "NA" << "\t" << "-";
        }
        cout << endl;
    }
}


void dijkstra(int numVertex, int *costMatrix, int src) {
    int* parent = (int*) malloc(numVertex * sizeof(int));
    int* distance = (int*) malloc(numVertex * sizeof(int));
    bool* visited = (bool*) malloc(numVertex * sizeof(bool));

    if (distance != NULL && parent != NULL && visited != NULL){
        fill(distance, distance + numVertex, INF);
        fill(visited, visited + numVertex, false);
        fill(parent, parent + numVertex, -1);

        distance[src] = 0;
        parent[src] = -1;

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
        printPath(numVertex, distance, parent);
    }   
}

/*
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

    dijkstra(numVertex, (int*)costMatrix, src);
}
*/