#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "utils.h"

using namespace std;

//__global__
//void bellmanFord(int numVertex, int src, int* costMatrix, int* distance, int* parent) {
//    distance[src] = 0;
//
//    for (int i = 0; i < numVertex - 1; i++) {
//        for (int u = 0; u < numVertex; u++) {
//            for (int v = 0; v < numVertex; v++) {
//                if (costMatrix[u * numVertex + v] != INF && distance[u] != INF && (distance[u] + costMatrix[u * numVertex + v]) < distance[v]) {
//                    parent[v] = u;
//                    distance[v] = distance[u] + costMatrix[u * numVertex + v];
//                }
//            }
//        }
//    }
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

    int numVertex, numEdges;
    map<int, list< pair<int, int > > > adjacencyList;

    fileToAdjacencyList(string("nyc-d.txt"), adjacencyList, numVertex, numEdges);
    // cout << adjacencyList.size() << " " << numVertex << " " << numEdges << endl;
}
