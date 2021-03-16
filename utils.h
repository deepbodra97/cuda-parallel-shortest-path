#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <sstream>

#define INF INT_MAX	

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

void fileToCostMatrix(string filename, int numVertex, int* costMatrix);
struct Graph* fileToAdjacencyList(string filename, struct Graph* costMatrix);
void fileToAdjacencyList(string filename, map<int, list<pair<int, int>>>& adjacencyList, int& numVertex, int& numEdges);

void adjacencyListToCSR(map<int, list<pair<int, int>>>& adjacencyList, vector<int>& vertices, vector<int>& indices, vector<int>& edges, vector<int>& weights);

void printPathSSSP(int numVertex, int* distance, int* parent);
void printPathAPSP(int numVertex, int* distance, int* parent);

#endif