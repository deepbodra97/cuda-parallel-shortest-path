#include "utils.h"

void printPathSSSP(int numVertex, int* distance, int* parent) {
    cout << "Node\tCost\tPath" << endl;
    for (int i = 0; i < numVertex; i++) {
        if (distance[i] != INF) {
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

void printPathAPSP(int numVertex, int* distance, int* parent) {
    for (int src = 0; src < numVertex; src++) {
        cout << "Source: " << src << endl;
        cout << "Node\tCost\tPath" << endl;
        for (int i = 0; i < numVertex; i++) {
            if (distance[src * numVertex + i] != INF) {
                cout << i << "\t" << distance[src * numVertex + i] << "\t";
                cout << i;

                int tmp = parent[src * numVertex + i];
                while (tmp != -1)
                {
                    cout << "<-" << tmp;
                    tmp = parent[src * numVertex + tmp];
                }
            }
            else {
                cout << i << "\t" << "NA" << "\t" << "-";
            }
            cout << endl;
        }
        cout << endl;
    }
}
