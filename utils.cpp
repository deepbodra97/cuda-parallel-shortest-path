#include "utils.h"

void splitBySpaceToVector(const string& line, vector<string>& tokens) {
    stringstream linestream(line);
    string token;
    while (linestream >> token) {
        tokens.push_back(token);
    }
}

void fileToCostMatrix(string filename, int* costMatrix, int& numVertex, int& numEdges) {
    cout << "fileToCostMatrix"<<endl;
    ifstream file(filename);
    string line;
    /*int skip = 8;
    while (skip != 0 && getline(file, line)) {
        skip--;
    }*/

    getline(file, line);
    vector<string> tokens;

    splitBySpaceToVector(line, tokens);
    numVertex = stoi(tokens[0]), numEdges = stoi(tokens[1]);
    while (getline(file, line)) {
        stringstream linestream(line);
        vector<string> tokens;
        string token;
        while (linestream >> token) {
            tokens.push_back(token);
        }
        int src = stoi(tokens[0]), dest = stoi(tokens[1]), cost = stoi(tokens[2]);
        cout <<"error"<< src << " " << dest << endl;
        costMatrix[src * numVertex + dest] = cost;
    }
    cout << "Finished reading input file" << endl;
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

void fileToAdjacencyList(string filename, map<int, list<pair<int, int>>>& adjacencyList, int& numVertex, int& numEdges) {
    cout << "Reading input file" << endl;
    ifstream file(filename);
    string line;
    /*int skip = 8;
    while (skip != 0 && getline(file, line)) {
        skip--;
    }*/

    getline(file, line);
    vector<string> tokens;

    splitBySpaceToVector(line, tokens);
    numVertex = stoi(tokens[0]), numEdges = stoi(tokens[1]);

    while (getline(file, line)) {
        tokens.clear();
        splitBySpaceToVector(line, tokens);
        int src = stoi(tokens[1]) - 1, dest = stoi(tokens[2]) - 1, cost = stoi(tokens[3]);
        // cout <<"error"<< src << " " << dest << endl;
        adjacencyList[src].push_back(make_pair(dest, cost));
    }
    cout << "Finished reading input file" << endl;
}

void adjacencyListToCSR(map<int, list<pair<int, int>>>& adjacencyList, vector<int>& vertices, vector<int>& indices, vector<int>& edges, vector<int>& weights) {
    int index = 0;
    indices.push_back(index);
    for (auto uIter = adjacencyList.begin(); uIter != adjacencyList.end(); ++uIter) {
        int u = uIter->first;
        vertices.push_back(u);
        index += uIter->second.size();
        indices.push_back(index);
        for (auto vIter = uIter->second.begin(); vIter != uIter->second.end(); ++vIter) {
            edges.push_back(vIter->first);
            weights.push_back(vIter->second);
        }
    }
}

void validateDistance(int numVertex, int* exp_distance, int* distance) {
    for (int i = 0; i < numVertex; i++) {
        assert(exp_distance[i] == distance[i]);
    }
    cout << "Validation Successful" << endl;
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
