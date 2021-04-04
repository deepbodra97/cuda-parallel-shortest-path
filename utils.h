#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <sstream>

#include <cassert>

#include <limits.h>

#define INF INT_MAX
#define THREADS_PER_BLOCK 1024

using namespace std;

struct AdjacencyListNode {
	int dest;
	int cost;
	struct AdjacencyListNode* next;
};

struct AdjacencyList {
	struct AdjacencyListNode* head;
};

struct Graph {
	int numVertex;
	struct AdjacencyList* neighbors;
};

int* fileToCostMatrix(string filename, int& numVertex, int& numEdges);
struct Graph* fileToAdjacencyList(string filename, struct Graph* costMatrix);
void fileToAdjacencyList(string filename, map<int, list<pair<int, int>>>& adjacencyList, int& numVertex, int& numEdges);

void adjacencyListToCSR(map<int, list<pair<int, int>>>& adjacencyList, vector<int>& vertices, vector<int>& indices, vector<int>& edges, vector<int>& weights);

void validateDistance(int numVertex, int* exp_distance, int* distance);

void printPathSSSP(int numVertex, int* distance, int* parent);
void printPathAPSP(int numVertex, int* distance, int* parent);

#endif