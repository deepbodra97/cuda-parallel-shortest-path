#ifndef UTILS_H
#define UTILS_H

#include<iostream>

#define INF INT_MAX

using namespace std;

void printPathSSSP(int numVertex, int* distance, int* parent);
void printPathAPSP(int numVertex, int* distance, int* parent);

#endif