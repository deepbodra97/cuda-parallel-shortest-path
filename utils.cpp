#include "utils.h"

void fileToCostMatrix(string filename, int numVertex, int* costMatrix) {
    cout << "fileToCostMatrix"<<endl;
    ifstream file(filename);
    string line;
    /*int skip = 8;
    while (skip != 0 && getline(file, line)) {
        skip--;
    }*/
    
    while (getline(file, line)) {
        stringstream linestream(line);
        vector<string> tokens;
        string token;
        while (linestream >> token) {
            tokens.push_back(token);
        }
        int src = stoi(tokens[1])-1, dest = stoi(tokens[2])-1, cost = stoi(tokens[3]);
        // cout <<"error"<< src << " " << dest << endl;
        costMatrix[src * numVertex + dest] = cost;
    }
}

struct AdjacencyListNode* newAdjacencyListNode(int dest, int weight) {
    struct AdjacencyListNode* newNode = (struct AdjacencyListNode*)malloc(sizeof(struct AdjacencyListNode));
    newNode->dest = dest;
    newNode->cost = weight;
    newNode->next = NULL;
    return newNode;
}

struct Graph* fileToAdjacencyList(string filename, struct Graph* graph) {
    ifstream file(filename);
    string line;
    /*int skip = 8;
    while (skip != 0 && getline(file, line)) {
        skip--;
    }*/

    while (getline(file, line)) {
        stringstream linestream(line);
        vector<string> tokens;
        string token;
        while (linestream >> token) {
            tokens.push_back(token);
        }

        int src = stoi(tokens[1]) - 1, dest = stoi(tokens[2]) - 1, cost = stoi(tokens[3]);

        struct AdjacencyListNode* newNode = newAdjacencyListNode(dest, cost);
        newNode->next = graph->neighbors[src].head;
        graph->neighbors[src].head = newNode;

        // cout << "error" << src << " " << dest << endl;
    }
    return graph;
}

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
