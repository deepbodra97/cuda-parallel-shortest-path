#include "utils.h"

void splitBySpaceToVector(const string& line, vector<string>& tokens) {
    stringstream linestream(line);
    string token;
    while (linestream >> token) {
        tokens.push_back(token);
    }
}

int* fileToCostMatrix(string filename, int& numVertex, int& numEdges) {
    cout << "Reading input file" << endl;
    ifstream file(filename);
    string line;

    getline(file, line);
    vector<string> tokens;

    splitBySpaceToVector(line, tokens);
    numVertex = stoi(tokens[0]), numEdges = stoi(tokens[1]);

    int* costMatrix = (int*)malloc(numVertex * numVertex * sizeof(int));
    if (costMatrix == NULL) {
        cout << "Malloc failed" << endl;
        throw std::exception();
    }
    fill(costMatrix, costMatrix + numVertex * numVertex, INF);

    while (getline(file, line)) {
        stringstream linestream(line);
        vector<string> tokens;
        string token;
        while (linestream >> token) {
            tokens.push_back(token);
        }
        int src = stoi(tokens[0]), dest = stoi(tokens[1]), cost = stoi(tokens[2]);
        costMatrix[src * numVertex + dest] = cost;
    }
    cout << "Finished reading input file" << endl;
    return costMatrix;
}

void fileToAdjacencyList(string filename, map<int, list<pair<int, int>>>& adjacencyList, int& numVertex, int& numEdges) {
    cout << "Reading input file" << endl;
    ifstream file(filename);
    string line;

    getline(file, line);
    vector<string> tokens;

    splitBySpaceToVector(line, tokens);
    numVertex = stoi(tokens[0]), numEdges = stoi(tokens[1]);

    while (getline(file, line)) {
        tokens.clear();
        splitBySpaceToVector(line, tokens);
        int src = stoi(tokens[0]), dest = stoi(tokens[1]), cost = stoi(tokens[2]);
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

void APSPInitDistanceParent(int numVertex, int* costMatrix, int* distance, int* parent) {
    cout << "Initializing distance and parent matrices using the cost matrix" << endl;
    for (int i = 0; i < numVertex; i++) {
        for (int j = 0; j < numVertex; j++) {
            if (i == j) {
                distance[i * numVertex + j] = 0;
                parent[i * numVertex + j] = -1;
            }
            else if (costMatrix[i * numVertex + j] == INF) {
                distance[i * numVertex + j] = INF;
                parent[i * numVertex + j] = -1;
            }
            else {
                distance[i * numVertex + j] = costMatrix[i * numVertex + j];
                parent[i * numVertex + j] = i;
            }
        }
    }
}

void validateDistanceSSSP(int numVertex, int* exp_distance, int* distance) {
    for (int i = 0; i < numVertex; i++) {
        assert(exp_distance[i] == distance[i]);
    }
    cout << "Validation Successful" << endl;
}

void validateDistanceAPSP(int numVertex, int* exp_distance, int* distance) {
    for (int i = 0; i < numVertex; i++) {
        for (int j = 0; j < numVertex; j++) {
            cout << i << " " << j << " " << exp_distance[i * numVertex + j] << " " << distance[i * numVertex + j] << endl;
            assert(exp_distance[i * numVertex + j] == distance[i * numVertex + j]);
        }
    }
    cout << "Validation Successful" << endl;
}

void printPathSSSP(int numVertex, int* distance, int* parent) {
    cout << "Node\tCost\tPath" << endl;
    for (int i = 0; i < numVertex; i++) {
        if (distance[i] != INF && distance[i] != 0) {
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

void writeOutPathSSSP(string filepath, int numVertex, int* distance, int* parent) {
    ofstream out(filepath);
    out << "Node\tCost\tPath" << endl;
    for (int i = 0; i < numVertex; i++) {
        if (distance[i] != INF && distance[i] != 0) {
            out << i << "\t" << distance[i] << "\t";
            out << i;

            int tmp = parent[i];
            while (tmp != -1)
            {
                out << "<-" << tmp;
                tmp = parent[tmp];
            }
            out << endl;
        }
        else {
            // uncomment this line to output "NA" for paths that don't exist
            // out << i << "\t" << "NA" << "\t" << "-";
            // out << endl;
        }
    }
    out.close();
}


void printPathAPSP(int numVertex, int* distance, int* parent) {
    for (int src = 0; src < numVertex; src++) {
        cout << "Source: " << src << endl;
        cout << "Node\tCost\tPath" << endl;
        for (int dest = 0; dest < numVertex; dest++) {
            if (distance[src * numVertex + dest] != INF && distance[src * numVertex + dest] != 0) {
                cout << dest << "\t" << distance[src * numVertex + dest] << "\t";
                cout << dest;

                int tmp = parent[src * numVertex + dest];
                while (tmp != -1)
                {
                    cout << "<-" << tmp;
                    tmp = parent[src * numVertex + tmp];
                }
                cout << endl;
            }
            else {
                // uncomment this line to output "NA" for paths that don't exist
                // cout << dest << "\t" << "NA" << "\t" << "-";
                // cout << endl;
            }
        }
        cout << endl;
    }
}

void writeOutPathAPSP(string filepath, int numVertex, int* distance, int* parent) {
    ofstream out(filepath);
    for (int src = 0; src < numVertex; src++) {
        out << "Source: " << src << endl;
        out << "Node\tCost\tPath" << endl;
        for (int dest = 0; dest < numVertex; dest++) {
            if (distance[src * numVertex + dest] != INF && distance[src * numVertex + dest] != 0) {
                out << dest << "\t" << distance[src * numVertex + dest] << "\t";
                out << dest;

                int tmp = parent[src * numVertex + dest];
                while (tmp != -1)
                {
                    out << "<-" << tmp;
                    tmp = parent[src * numVertex + tmp];
                }
                out << endl;
            }
            else {
                // uncomment this line to output "NA" for paths that don't exist
                // out << dest << "\t" << "NA" << "\t" << "-";
                // out << endl;
            }
        }
        out << endl;
    }
    out.close();
}