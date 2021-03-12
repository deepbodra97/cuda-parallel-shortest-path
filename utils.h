#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include <sstream>

#define INF INT_MAX

using namespace std;

void fileToCostMatrix(string filename, int numVertex, int* costMatrix);
void printPathSSSP(int numVertex, int* distance, int* parent);
void printPathAPSP(int numVertex, int* distance, int* parent);

#endif