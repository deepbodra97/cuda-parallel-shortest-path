#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
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
void printPathSSSP(int numVertex, int* distance, int* parent);
void printPathAPSP(int numVertex, int* distance, int* parent);

#endif