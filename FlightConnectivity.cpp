#include <iostream>
#include <string>
#include <vector>   
#include <fstream>  // For reading the csv
#include <iomanip>  // Input/output manipulators for formatted output
#include <stdio.h>
#define INT_MAX 10000
using namespace std;

ifstream file("airports.csv"); // Accessing the file

// An object of the Edge class represents a flight on the graph which holds the
// origin, destination, distance, and cost info of the flight
class Edge {
public:
    // Constructor
    Edge(int origin = 0, int destination = 0, int distance = 0, int cost = 0) : 
        origin(origin), destination(destination), distance(distance), cost(cost) {}


    bool operator<(const Edge& other) const {
        if(this->cost < other.cost) {
            return true;
        }

        return false;
    }

    int origin;
    int destination;
    int distance;
    int cost;
};

// An object of the Vertex class represents an airport
class Vertex {
public:
    string airport;
    string location;
    bool visited;

    Vertex(){}

    //constructor
    Vertex(string airport, string location){
        this->airport = airport;
        this->location = location;
    };

    bool getVisited() const {return visited; } // Returns true if an aiport has been visited
    void setVisited(bool v) { visited = v; } // Setter function for whether an aiport has been visited or not
    string getState(){
        return location.substr(location.find(",")+2);
    }
};

// Breadth-First Search helper class to store data and minimize the number of operations
class BFShelper {
public:
    int vertex;       
    int stops;        
    vector<int> path;
    int distance;
    
    bool operator<(const BFShelper& other) const {
        if(this->distance < other.distance) {
            return true;
        }
        return false;
    }
};

// Struct for airport connections to simplify printing for task 5
struct AirportConnections {
    string airport_name;
    int totalConnections;
};

// Extra edge class without distance
class noDistEdge {
public:
    noDistEdge(int origin = 0, int destination = 0, int cost = 0) : 
        origin(origin), destination(destination), cost(cost) {}


    bool operator<(const Edge& other) const {
        if(this->cost < other.cost) {
            return true;
        }

        return false;
    }

    int origin;
    int destination;
    int cost;
};

// Creates a disjoint set to unify subsets together so it can be sorted and accessed
class DisjointSet {
private:
    vector<int> parent;
    vector<int> rank;

public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unite_kruskal(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }
};

// Min Heap class
class MinHeap {
public:
    vector<Edge> data;
    MinHeap() {}
    MinHeap(std::vector<Edge> v){
        this->data = v;
    }

    void insert(const Edge& val){
        data.push_back(val);
        int n = data.size() - 1; // Index of the last node
        //percolate UP
        while (data[n].distance < data[(n - 1) / 2].distance) {
        swap(data[n], data[(n - 1) / 2]);
        n = (n - 1) / 2; // Makes the current index equal to the parent index
        }
    }

    bool isEmpty() const {
        return data.empty();
    }

    Edge delete_min(){   
        Edge res = data[0];
        data[0] = data[data.size() - 1]; // Set the root with the value of the last node
        data.pop_back(); // Deletes the last node

        // Percolate down
        percolate_down(0);

        return res;
    }

private:
    void swap(Edge& v1, Edge& v2) {
        Edge tmp = v1;
        v1 = v2;
        v2 = tmp;
    }

    void percolate_down(int i){
        if (data.empty() || i >= data.size() || i < 0) {
        return;
        }

        int parent_index = i;
        int kids_min_index = i;

        do {
            if (parent_index * 2 + 1 >= data.size()) {
                break;
            }
            else if (parent_index * 2 + 2 < data.size()){ // Has 2 children
                kids_min_index = min_index(parent_index * 2 + 1, parent_index * 2 + 2);

            }
            else if (parent_index * 2 + 1 < data.size()) { // Has left child
                kids_min_index = parent_index * 2 + 1;
            }
            // Check is the smallest child is smaller than the parent
            if (data[kids_min_index] < data[parent_index]) {
                swap(data[parent_index], data[kids_min_index]);
                parent_index = kids_min_index;
            }
            else {
                break;
            }

        } while(1);

        }
    int min_index(int i1, int i2) const{
        if (i1 >= data.size() || i2 >= data.size() || i1 < 0 || i2 < 0) {
            throw std::string ("min_index: incorrect index");
        }

        return (data[i1] < data[i2] ? i1 : i2);
    }
};

// Queue class
template<typename T>
class Queue {
private:
    std::vector<T> queue;

public:
    Queue() {}

    bool isEmpty() const {
        return queue.empty();
    }

    // Enqueue an element to the back of the queue
    void enqueue(const T& q) {
        queue.push_back(q);
    }

    // Dequeue an element 
    T dequeue() {
        if (isEmpty()) {
            throw std::out_of_range("Queue is empty");
        }
        T front = queue.front();
        queue.erase(queue.begin());
        return front;
    }
};

// Undirected Graph class
class UndirectedGraph {
public:
    vector<Vertex> vertices;
    int numVertices;
    int get_vertex_index(const string& ver);
    int get_vertex_index(const Vertex& ver);
    void insert_vertex(const Vertex& ver);
    void addEdge(string u, string v, int cost) {
        int i1 = get_vertex_index(u);
        int i2 = get_vertex_index(v);
        noDistEdge originEdge(i1, i2, cost);
        noDistEdge destEdge(i2, i1, cost);
        noDistEdges[i1].push_back(originEdge);
        noDistEdges[i2].push_back(destEdge);
    }

    // Prints the undirected graph
    void printGraph() {
        std::cout << "Undirected Graph G_u:\n";
        for (const auto& edge : noDistEdges) {
            int u = edge[0].origin;
            int v = edge[0].destination;
            int cost = edge[0].cost;
            std::cout << "Vertex " << vertices[u].airport << " connects to Vertex " << vertices[v].airport << " with cost " << cost << endl;
        }
    }

    // Sorting algorithm
    void selectionSort(vector<noDistEdge>& edges) {
        int n = edges.size();
        for (int i = 0; i < n - 1; ++i) {
            int minIndex = i;
            for (int j = i + 1; j < n; ++j) {
                if (edges[j].cost < edges[minIndex].cost) {
                    minIndex = j;
                }
            }
            if (minIndex != i) {
                swap(edges[i], edges[minIndex]);
            }
        }
    }

    // Kruskal's algorithm for MST
    void kruskal() {
        vector<noDistEdge> mstEdges;
        vector<noDistEdge> flattenedEdges;
        numVertices = vertices.size();
        DisjointSet ds(numVertices);
        int mstWeight = 0;

        // Flattenes edges
        for (const auto& edges : noDistEdges) {
            for (const auto& edge : edges) {
                flattenedEdges.push_back(edge);
            }
        }

        // Call the sorting algorithm
        selectionSort(flattenedEdges);
        for (const auto& edge : flattenedEdges) {
            int u = edge.origin;
            int v = edge.destination;
            int setU = ds.find(u);
            int setV = ds.find(v);
            if (setU != setV) {
                ds.unite_kruskal(setU, setV);
                mstWeight += edge.cost;
                mstEdges.push_back(edge);
            }
        }

        std::cout << "Minimal Spanning Tree Edges:\n";
        for (const auto& edge : mstEdges) {
            std::cout << vertices[edge.origin].airport << " - " << vertices[edge.destination].airport << " : " << edge.cost << '\n';
        }
        std::cout << "Total MST Weight: " << mstWeight << '\n';
    }

    // Prim's algorithm for MST
    void prim() {
        int numVertices = vertices.size();
        vector<bool> visited(numVertices, false); 
        vector<int> parent(numVertices, -1);      
        vector<int> costs(numVertices, INT_MAX); 
        costs[0] = 100;

        // Repeat until all vertices are visited
        for (int i = 0; i < numVertices - 1; i++) {
            // Locate the vertex with the minimum cost so we can start there.
            int minCost = INT_MAX;
            int minVertex = -1;
            for (int j = 0; j < costs.size(); j++) {
                if (!visited[j] && costs[j] < minCost) {
                    minCost = costs[j];
                    minVertex = j;
                }
            }
            int u = minVertex;
            if (u == -1) {
                // If u is -1, then graph is disconnected. so it cant create one.
                std::cout << "DISCONNECTED GRAPH: Minimal Spanning Tree cannot be formed!\n";
                return;
            }
            visited[u] = true; // Mark the current vertex as visited

            // Update costs and parents of adjacent vertices
            for (const auto& edge : noDistEdges[u]) {
                int v = edge.destination;
                int cost = edge.cost;
                if (!visited[v] && cost < costs[v]) {
                    parent[v] = u;
                    costs[v] = cost;
                }
            }
        }
        // If the graph is connected, output the MST
        std::cout << "Minimal Spanning Tree Edges:\n";
        int totalCost = 0;
        for (int i = 1; i < numVertices; ++i) {
            std::cout << vertices[parent[i]].airport << " - " << vertices[i].airport << " : " << costs[i] << '\n';
            totalCost += costs[i];
        }
        std::cout << "Total MST Weight: " << totalCost << '\n';
    }

private:
    vector<vector<noDistEdge>> noDistEdges; // Vector of vectors representing edges (u, v, cost)
};

// Weighted-Directed Graph class
class WDG {
public:
    WDG() {}
    std::vector<Vertex> vertices; // Vertices
    std::vector<vector<Edge>> edges; // Connections
    int get_vertex_index(const Vertex& ver);
    int get_vertex_index(const string& ver);
    void insert_vertex(const Vertex& ver);
    void add_edge(const Vertex& ver1, const Vertex& ver2, int distance, int cost); // Connect ver1 with ver2
    void display_connections(); // Displays all connections
    void reset_visited(); // Sets the values of the visited vertices to false
    void dijkstra_shortest_state(string origin_airport, string destination_state); 
    void dijkstra_shortest_path_helper(string origin_airport, string destination_airport);
    vector<string> dijkstra_shortest_path(string origin_airport, string destination_airport);
    vector<string> shortest_path_with_stops(string origin_airport, string destination_airport, int max_stops);
    vector<int> BFS_shortest_path(string origin_airport, string destination_airport, int stops);
};

// Simple swap function for airport connections to assist with sorting
void swap(AirportConnections &a, AirportConnections &b) {
    AirportConnections temp = a;
    a = b;
    b = temp;
}

// Grabs vertex index from its location in the vertices vector
int UndirectedGraph::get_vertex_index(const string & ver) {
    for(int i = 0; i < vertices.size(); i++) {
        if (vertices[i].airport == ver) {
            return i;
        }
    }
    return -1;
}

// Insert vertex function for the undirected graph class
void UndirectedGraph::insert_vertex(const Vertex& ver) {
    if (get_vertex_index(ver) == -1) {
        vertices.push_back(ver); // Insert the vertex to the array of vertices
        std::vector<noDistEdge> tmp;
        noDistEdges.push_back(tmp); // Insert empty placeholder to the nodistedges
    }
}

// Grabs vertex index from its location in the vertices vector
int UndirectedGraph::get_vertex_index(const Vertex& ver) {
    for(int i = 0; i < vertices.size(); i++) {
        if (vertices[i].airport == ver.airport) {
            return i;
        }
    }
    return -1;
}

UndirectedGraph createUndirectedGraph(WDG& directedGraph) {
    UndirectedGraph G_u;
    for(int i = 0; i < directedGraph.vertices.size(); i++){
            Vertex u(directedGraph.vertices[i].airport, directedGraph.vertices[i].location);
            G_u.insert_vertex(u); // Origin 
        }

    // Iterate through each vertex and its adjacent vertices
    for (const auto& vertex : directedGraph.vertices) {
        string u = vertex.airport;
        for (const auto& edge : directedGraph.edges[directedGraph.get_vertex_index(vertex.airport)]) {
            string v = directedGraph.vertices[edge.destination].airport;
            int cost = edge.cost; 

            // Check if there is a reverse edge
            bool hasReverseEdge = false;
            int idx_v = directedGraph.get_vertex_index(v);
            for (const auto& revEdge : directedGraph.edges[idx_v]) {
                if (directedGraph.vertices[revEdge.destination].airport == u) {
                    hasReverseEdge = true;
                    // Use lower cost of reverse edge
                    if (revEdge.cost < cost) {
                        cost = revEdge.cost;
                    }
                    break;
                }
            }
            if (!hasReverseEdge) {
                G_u.addEdge(u, v, cost);
            } else {
                G_u.addEdge(v, u, cost);
            }
        }
    }
    return G_u;
}

vector<int> WDG::BFS_shortest_path(string origin_airport, string destination_airport, int stops) {
    int i_origin = get_vertex_index(origin_airport);
    vector<BFShelper> tmp;
    Queue<BFShelper> q;
    BFShelper origin = {i_origin, -1, {i_origin}, 0}; // Start with origin 
    q.enqueue(origin);
    reset_visited();

    while (!q.isEmpty()) {
        BFShelper item = q.dequeue();
        if (vertices[item.vertex].airport == destination_airport && item.stops == stops) { 
            // If destination is met with stops being equal to the value of passed stops, then push into a comparison section.
            tmp.push_back(item);
        }
        // If it excedes the amount of stops, then stops.
        if (item.stops > stops) {
            continue;
        }

        if(vertices[item.vertex].airport != destination_airport){
            vertices[item.vertex].setVisited(true);
        }
        // Loops through edges and pushes neighbors into the queue.
        for (const Edge& e : edges[item.vertex]) {
            if (!vertices[e.destination].getVisited()) {
                vector<int> newPath = item.path;
                newPath.push_back(e.destination);
                BFShelper temp = {e.destination, item.stops + 1, newPath, e.distance};
                q.enqueue(temp);
            }
        }
    }
    reset_visited();
    if(tmp.size() == 0){
        return vector<int>(); // No path found
    }else{
        // Determines object with the smallest distance and passes through that.
        BFShelper shortest = tmp[0]; 
        for(int i = 0; i < tmp.size(); i++){
            if(tmp[i].distance < shortest.distance){
                shortest = tmp[i];
            }
        }
        return shortest.path;
    }
}

vector<string> WDG::shortest_path_with_stops(string origin_airport, string destination_airport, int stops) {
    vector<int> path;
    try {
        // Sets path to the shortest path found from bfs
        path = BFS_shortest_path(origin_airport, destination_airport, stops);
    } catch (std::logic_error& e) {
        throw e;
    }

    // If path is empty, output none
    vector<string> output;
    if (path.empty()) {
        output.push_back("no path");
        std::cout << "Shortest route from " << origin_airport << " to " << destination_airport << " with " << stops << " stops: None" << endl;
        return output;
    }
    // Calculate total cost and distance and concatenate path.
    int totalCost = 0;
    int distance = 0;
    string pathing;
    for (int i = 0; i < path.size(); ++i) {
        int vertex_index = path[i];
        pathing += (i == 0 ? "" : "->") + vertices[vertex_index].airport;
        if (i > 0) {
            int prev_vertex_index = path[i - 1];
            for (const auto& edge : edges[prev_vertex_index]) {
                if (edge.destination == vertex_index) {
                    totalCost += edge.cost;
                    distance += edge.distance;
                    break;
                }
            }
        }
    }

    output.push_back(pathing);
    output.push_back(to_string(totalCost));
    output.push_back(to_string(distance));

    // Manage the output based off of size
    std::cout << "Shortest route from " << origin_airport << " to " << destination_airport << " with " << stops << " stops: ";
    if (output.size() == 1) {
        std::cout << output[0] << std::endl;
    } else if (output.size() == 3) {
        std::cout << output[0] << ". The cost is " << output[1] << ". The distance is " << output[2] << ".\n";
    } else {
        std::cout << std::endl;
    }
    return output;
}

void WDG::insert_vertex(const Vertex& ver) {
    if (get_vertex_index(ver) == -1) {
        vertices.push_back(ver); // Insert the vertex to the array of vertices
        std::vector<Edge> tmp;
        edges.push_back(tmp); // Insert empty placeholder to the edges
    }
}

int WDG::get_vertex_index(const Vertex& ver) {
    // Returns the index of the vertex inside of the vertices vector
    for(int i = 0; i < vertices.size(); i++) {
        if (vertices[i].airport == ver.airport) {
            return i;
        }
    }
    return -1;
}

int WDG::get_vertex_index(const string & ver) {
    // Returns the vertex index from the vertices vector, but this time taking a string
    for(int i = 0; i < vertices.size(); i++) {
        if (vertices[i].airport == ver) {
            return i;
        }
    }

    return -1;
}

void WDG::reset_visited() {
    // Sets all visited parts of the vertexes in vertices to false
    for(Vertex& v : vertices) {
        v.setVisited(false);
    }
}

vector<string> WDG::dijkstra_shortest_path(string origin_airport, string destination_airport) {
  vector<string> output;
  string temp = "";
  temp.append(origin_airport);
  temp.append(" -> ");
  int i_origin, i_dest=0;
  i_origin = get_vertex_index(origin_airport);
  i_dest = get_vertex_index(destination_airport);
  reset_visited();
  std::vector<int> distances(vertices.size());
  std::vector<int> costs(vertices.size());
  for(int i = 0; i < vertices.size(); i++){
        distances[i] = (i == i_origin) ? 0 : INT_MAX;
        costs[i] = (i == i_origin) ? 0 : INT_MAX;
    }
  vector<int> prev(vertices.size(), -1);
  MinHeap heap;
  int vertices_visited = 0;
  int cur_ver = i_origin;

  // If heap is not empty or the vertices visited is less then the size of the vertices vector.
  while(!heap.isEmpty() || vertices_visited < vertices.size())
    {
      for(int j = 0; j < edges[cur_ver].size(); j++)
        {
          int i_adjacent_ver = edges[cur_ver][j].destination;
          if(vertices[i_adjacent_ver].getVisited() == false)
          {
            heap.insert(edges[cur_ver][j]);
            int dist_from_source = distances[cur_ver] + edges[cur_ver][j].distance;
            int cost_from_source = costs[cur_ver] + edges[cur_ver][j].cost;
            if(dist_from_source < distances[i_adjacent_ver])
            {
              distances[i_adjacent_ver] = dist_from_source;
              costs[i_adjacent_ver] = cost_from_source;
            }
          }
        }
        // Sets e to the minimum value deleted from the heap.
        Edge e = heap.delete_min();
        cur_ver = e.destination;
        vertices[cur_ver].setVisited(true);
        vertices_visited++;
        if(vertices[cur_ver].airport == destination_airport)
        {
            vector<int> shortest_path, cost_of_path;
            int v = cur_ver;
            while(v != -1)
            {
                shortest_path.push_back(v);
                cost_of_path.push_back(costs[v]);
                v=prev[v];
            }
            for(int i = shortest_path.size() - 1; i >= 0; i--){
                temp.append(vertices[shortest_path[i]].airport);
                if(i>0){temp.append(" -> ");}
            }
        reset_visited();
        output.push_back(temp);
        output.push_back(to_string(distances[cur_ver]));
        output.push_back(to_string(costs[cur_ver]));
        return output;
        }
        for(int j = 0; j < edges[cur_ver].size(); j++)
            {
            int i_adjacent_ver = edges[cur_ver][j].destination;
            int i_adj_ver_dist = edges[cur_ver][j].distance;
            int i_adj_ver_cost = edges[cur_ver][j].cost;
            if(vertices[i_adjacent_ver].getVisited() == false)
            {
                int dist_from_source = distances[cur_ver] + i_adj_ver_dist;
                int cost_from_source = costs[cur_ver] + i_adj_ver_cost;
                if(dist_from_source < distances[i_adjacent_ver])
                {
                distances[i_adjacent_ver] = dist_from_source;
                costs[i_adjacent_ver] = cost_from_source;
                prev[i_adjacent_ver] = cur_ver;
                heap.insert(edges[cur_ver][j]);
                }
            }
        }
        // If heap is empty and it never visits the destination, then state no path exists.
        if(heap.isEmpty() && !vertices[i_dest].getVisited())
        {
            output.clear();
            output.push_back("None");
            return output;
        }
        }
    reset_visited();
    return output;
}

void WDG::add_edge(const Vertex& ver1, const Vertex& ver2, int distance, int cost) {
    // Links vertices together with an edge
    // Creates the edge using the vertices and then pushes it into the vector
    int i1 = get_vertex_index(ver1);
    int i2 = get_vertex_index(ver2);
    if (i1 == -1 || i2 == -1) {
        throw std::string("Add_edge: incorrect vertices");
    }
    Edge v(i1, i2, distance, cost);
    edges[i1].push_back(v);
}

void WDG::display_connections(){
    int numOfVertices = vertices.size();    
    vector<AirportConnections> connections(numOfVertices);
    // Calculate total connections for each airport
    for(int i = 0; i < numOfVertices; i++) {
        connections[i].airport_name = vertices[i].airport;
        int inbound = 0;
        int outbound = 0;
        
        // Inbound
        for(int j = 0; j < edges.size(); j++) {
            for(int k = 0; k < edges[j].size(); k++) {
                if(connections[i].airport_name == vertices[edges[j][k].destination].airport) {
                    inbound++;
                }
            }
        }
        // Outbound
        for(int j = 0; j < edges[i].size(); j++) {
            if(connections[i].airport_name == vertices[edges[i][j].origin].airport) {
                outbound++;
            }
        }
        // Total connections
        connections[i].totalConnections += inbound;
        connections[i].totalConnections += outbound;
    }

    // Bubble sort algorithm for airport connections based on connection count
    for(int i = 0; i < connections.size() - 1; i++) {
        for(int j = 0; j < connections.size() - i - 1; j++) {
            if(connections[j].totalConnections < connections[j + 1].totalConnections) {
                swap(connections[j], connections[j + 1]);
            }
        }
    }
    // Print statement
    std::cout << "Airport\t\t\tConnections" << endl;
    for(int i = 0; i < connections.size(); i++) {
        std::cout << connections[i].airport_name << "\t\t\t" << connections[i].totalConnections << endl;
    }
}

void WDG::dijkstra_shortest_path_helper(string origin_airport, string destination_airport){
    // Assists with calling the dijkstras shortest path and printing the outputs
    vector<string> pathInfo = dijkstra_shortest_path(origin_airport, destination_airport);
    std::cout << "Shortest route from " << origin_airport << " to " << destination_airport << ": " << pathInfo[0];
    if(pathInfo.size()==1)
    {
        std::cout<< std::endl;
        return;
    }
    else
    {
        std::cout << ". The length is " << pathInfo[1] << ". The cost is " << pathInfo[2] << ".\n";
    }
}

void WDG::dijkstra_shortest_state(string origin_airport, string destination_state)
{
// Iterates through all vertices, if destination state is the same as the vertices, then compute dijkstra's shortest path on it
    std::cout << "Shortest paths from " << origin_airport << " to " << destination_state <<" state airports are: \n";
    std::cout << "Path\t" << "Length\t" << "Cost\n";
    for(int i = 0; i < vertices.size(); i++)
    {
        if(destination_state == vertices[i].getState())
        {
            vector<string> temp = dijkstra_shortest_path(origin_airport, vertices[i].airport);
            if(temp.size() != 1)
            {
                std::cout << temp[0] << "\t" << temp[1] << "\t" << temp[2] << "\n";
            }
        }
    }
}

vector<string> fileInfo(string line){
    vector<string> information;
    string origin_airport, dest_airport, origin_location, dest_location;
    string distance, cost;
    int temp;

    origin_airport = line.substr(0,3);
    dest_airport = line.substr(4,3);

    line = line.substr(9);
    temp = line.find('"');
    origin_location = line.substr(0,temp);

    line = line.substr(temp+3);
    temp = line.find('"');
    dest_location = line.substr(0,temp);

    line = line.substr(temp+2);
    temp = line.find(',');
    distance = line.substr(0,temp);
    cost = line.substr(temp+1);

    information.push_back(origin_airport);
    information.push_back(origin_location);
    information.push_back(dest_airport);
    information.push_back(dest_location);
    information.push_back(distance);
    information.push_back(cost);
    return information;
}

int main(){
    WDG G; // Creates a weighted directed graph G

    // Skips the first line of the file containing the CSV format
    string skipheader, fileLine;
    getline(file, skipheader);

    cout << "Task #1: Using dataset (airports.csv), Create a weighted directed graph G" << endl;
    while(getline(file, fileLine)){
        vector<string> line = fileInfo(fileLine); // Read a line from airports.csv
        Vertex v1(line[0], line[1]); // Create a vertex v1 with identifier and additional info
        Vertex v2(line[2], line[3]); // Create a vertex v2 with identifier and additional info
        int distance = stoi(line[4]); // Convert the distance from a string to an integer
        int cost = stoi(line[5]); // Convert the cost from a string to an integer
        G.insert_vertex(v1); // Insert v1 into the graph
        G.insert_vertex(v2); // Insert v2 into the graph
        G.add_edge(v1, v2, distance, cost); // Add an edge between v1 and v2 with a distance and a cost
    }   
    cout << "\n------------------------ --------------------------------------------\n\n";

    // Task 2
    cout << "Task #2: Find the shortest path between the given origin airport and a destination airport\n" << endl;
    G.dijkstra_shortest_path_helper("JFK", "TPA");
    G.dijkstra_shortest_path_helper("GNV", "MIA");
    cout << "\n--------------------------------------------------------------------\n\n";

    // Task 3
    cout << "Task #3: Find all shortest paths between a given origin airport and all airports in the given destination state\n" << endl;
    G.dijkstra_shortest_state("AUS", "FL");
    cout << "\n--------------------------------------------------------------------\n\n";

    // Task 4
    cout << "Task #4: Shortest path between a given origin airport and a destination airport with a given number of stops\n" << endl;
    G.shortest_path_with_stops("BWI", "TPA", 1);
    G.shortest_path_with_stops("BWI", "TPA", 2);
    G.shortest_path_with_stops("BWI", "TPA", 3);
    G.shortest_path_with_stops("BWI", "TPA", 4);
    cout << "\n--------------------------------------------------------------------\n\n";

    // Task 5
    cout << "Task #5: Count and Display Total Direct Inbound and Outbound Connections\n" << endl;
    G.display_connections();
    cout << "\n--------------------------------------------------------------------\n\n";

    // Task 6
    cout << "Task #6: Create An Undirected Graph G_u\n" << endl;
    UndirectedGraph G_u = createUndirectedGraph(G); // Creates an undirected graph G_u using the WDG G as a parameter
    G_u.printGraph();
    cout << "\n--------------------------------------------------------------------\n\n";

    // Task 7
    cout << "Task #7: Prim's Algorithm to create a MST\n" << endl;
    G_u.prim();
    cout << "\n--------------------------------------------------------------------\n\n";

    // Task 8
    cout << "Task #8: Kruskal's Algorithm to create a MST\n" << endl;
    G_u.kruskal();
    cout << "\n--------------------------------------------------------------------\n\n";

}