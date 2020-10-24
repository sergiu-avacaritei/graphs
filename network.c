// Extract information about a given network.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>

#define MAX_DIGITS_INT    10
#define MAX_DIGITS_DOUBLE 317

typedef struct node {
    int key;
    double weight;
    struct node *next;
}node;

typedef struct graph {
    int numNodes;
    bool *visited;
    bool *isolated;
    bool *nodes;
    struct node **neighbours;
}graph;

//------------------------------------------------------------------------------
// ####################### UTILITY FUNCTIONS FOR NETWORK #######################
//------------------------------------------------------------------------------

// Create a new empty graph.
graph *newGraph() {
    graph *g = malloc(sizeof(graph));
    *g = (graph) {0, NULL, NULL, NULL, NULL};
    return g;
}

// Create a new node.
node *newNode(int key, double weight) {
    node *newNode = malloc(sizeof(node));
    *newNode = (node) {key, weight, NULL};
    return newNode;
}

// Add a new edge to network.
// Note: if to = -1, then it adds an isolated node to network.
void addEdge(graph *network, int from, int to, double weight) {
    if (to != -1) {
        node *toNode = newNode(to, weight);
        toNode->next = network->neighbours[from];
        network->neighbours[from] = toNode;
        network->nodes[from] = true;
        network->nodes[to] = true;
    }
    else {
        node *isolatedNode = newNode(from, weight);
        network->neighbours[from] = isolatedNode;
        network->nodes[from] = true;
        network->isolated[from] = true;
    }
}

// Expand the network, reallocating memory and initializing every field of the
// struct graph.
void expandNetwork(graph *g, int numNodes) {
    g->visited = realloc(g->visited, numNodes * sizeof(bool));
    g->isolated = realloc(g->isolated, numNodes * sizeof(bool));
    g->nodes = realloc(g->nodes, numNodes * sizeof(bool));
    g->neighbours = realloc(g->neighbours, numNodes * sizeof(node*));
    for (int i = g->numNodes; i < numNodes; i++) {
        g->visited[i] = false;
        g->isolated[i] = false;
        g->nodes[i] = false;
        g->neighbours[i] = NULL;
    }
    g->numNodes = numNodes;
}

// Destroy the network.
void destroyNetwork(graph *network) {
    for (int i = 0; i < network->numNodes; i++) {
        node* neighbour = network->neighbours[i];
        while (neighbour) {
            node *nextNeighbour = neighbour->next;
            free(neighbour);
            neighbour = nextNeighbour;
        }
    }
    free(network->neighbours);
    free(network->visited);
    free(network->isolated);
    free(network->nodes);
    free(network);
}


// Check if a network is empty.
bool emptyNetwork(graph *network) {
    return network->numNodes == 0;
}

// Check if a node is isolated.
bool isolated(graph *network, int node) {
    return network->isolated[node] == true;
}

// Check if a node is in the network.
bool hasNode(graph *network, int node) {
    return network->nodes[node] == true;
}

// Check if a node has at least one neighbour.
bool hasNeighbours(graph *network, int node) {
    return (network->neighbours[node] != NULL && ! isolated(network, node));
}

// Chech if the node "n" is amongst "currentNode"`s neighbours.
bool isNeighbour(graph *network, int currentNode, int n) {
    if (isolated(network, currentNode)) return false;
    node *neighbour = network->neighbours[currentNode];
    while (neighbour) {
        if (neighbour->key == n) return true;
        neighbour = neighbour->next;
    }
    return false;
}

// Check if a node is linked to itself.
bool linkedToItself(graph *network, int node) {
    if (isolated(network, node)) return false;
    else if (hasNeighbours(network, node)) {
        if (network->neighbours[node]->key == node) return true;
    }
    return false;
}

// Check if a node is visited.
bool visited(graph *network, int node) {
    return network->visited[node] == true;
}

// Check if all nodes are visited.
bool visitedAll(graph *network) {
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i) && ! visited(network, i)) return false;
    }
    return true;
}

// Count the number of neighbours of a given node n.
int countNeighbours(graph *network, int n) {
    if (isolated(network, n)) return 0;
    int counter = 0;
    node *neighbour = network->neighbours[n];
    while (neighbour) {
        counter++;
        neighbour = neighbour->next;
    }
    return counter;
}

// Set all nodes as not visited.
void initializeVisited(graph *network) {
    for (int i = 0; i < network->numNodes; i++) network->visited[i] = false;
}

// Find the in degree of all nodes in the network.
// E.g.: inDegree[i] = 3, means that the node "i" has in degree equal to 3.
int *initializeInDegree(graph *network) {
    int *inDegree = malloc(network->numNodes * sizeof(int));
    for (int i = 0; i < network->numNodes; i++) inDegree[i] = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i)) {
            node *currentNode = network->neighbours[i];
            if (! isolated(network, i)) {
                while (currentNode != NULL) {
                    inDegree[currentNode->key]++;
                    currentNode = currentNode->next;
                }
            }
        }
    }
    return inDegree;
}

// Find the out degree of all nodes in the network.
// E.g.: outDegree[i] = 3, it means that the node "i" has out degree equal to 3.
int *initializeOutDegree(graph *network) {
    int *outDegree = malloc(network->numNodes * sizeof(int));
    for (int i = 0; i < network->numNodes; i++) outDegree[i] = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNeighbours(network, i) && ! isolated(network, i)) {
            outDegree[i] = countNeighbours(network, i);
        }
    }
    return outDegree;
}

// (For Dijkstra`s algorithm)
// Set the distances to all of the nodes equal to DBL_MAX.
double *initializeDistance(graph *network) {
    double *distance = malloc(network->numNodes * sizeof(double));
    for (int i = 0; i < network->numNodes; i++) distance[i] = DBL_MAX;
    return distance;
}

// Do a depth-first search traversal, marking visited nodes as visited (true).
void DFS(graph *network, int currentNode) {
    network->visited[currentNode] = true;
    node *neighbour = network->neighbours[currentNode];
    while (neighbour) {
        if (! visited(network, neighbour->key)) DFS(network, neighbour->key);
        neighbour = neighbour->next;
    }
}

// Reverse the network. Return a new network on the same set of nodes with all
// the edges reversed with respect to the orientation of the corresponding edges
// in the given network. Also known as "transpose graph" in graph theory.
// I.e.: For every nodes x,y the edge x-y is changed to y-x.
graph *reverseNetwork(graph *network) {
    graph *revNetwork = newGraph();
    expandNetwork(revNetwork, network->numNodes);
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i)) {
            node *neighbour = network->neighbours[i];
            if (! network->isolated[i]) {
                while (neighbour) {
                    addEdge(revNetwork, neighbour->key, i, neighbour->weight);
                    neighbour = neighbour->next;
                }
            }
            else addEdge(revNetwork, neighbour->key, -1, -1);
        }
    }
    return revNetwork;
}

// Return the maximum of two integers.
int max(int x, int y) {
    return x > y ? x : y;
}

//------------------------------------------------------------------------------
// #################### EXTRACTING INFORMATION FROM NETWORK ####################
//------------------------------------------------------------------------------

// Print the representation of network.
void printNetwork(graph *network) {
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i)) {
            node* neighbour = network->neighbours[i];
            printf("%d: ", i);
            while (neighbour) {
                printf("[%d | %lf] ", neighbour->key, neighbour->weight);
                if (isolated(network, neighbour->key)) printf("(ISOLATED) ");
                printf("-> ");
                neighbour = neighbour->next;
            }
            printf("\n");
        }
    }
}

// Count the number of nodes. Iterate through all nodes in the network and
// count them up.
int countNodes(graph *network) {
    int counter = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i)) counter++;
    }
    return counter;
}

// Count the number of edges. Iterate through all nodes in the network and add
// up their number of neighbours to the "counter" variable.
int countEdges(graph *network) {
    int counter = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i) && ! isolated(network, i)) {
            counter += countNeighbours(network, i);
        }
    }
    return counter;
}

// Check if the network is strongly connected.
// I.e.: Do 2 DFS traversals, one on the given network and one on the reversed
// network, starting at the same node. After that, if there still are nodes that
// were not visited, the network is not strongly connected. Otherwise, the
// network is strongly connected.
bool stronglyConnected(graph *network) {
    graph *revNetwork = reverseNetwork(network);
    initializeVisited(network);
    initializeVisited(revNetwork);
    bool ok = true;
    int startNode = -1;
    for (int i = 0; i < network->numNodes && startNode == -1; i++) {
        if (hasNode(network, i)) startNode = i;
    }
    DFS(network, startNode);
    DFS(revNetwork, startNode);
    if (! visitedAll(network) || ! visitedAll(revNetwork)) ok = false;
    destroyNetwork(revNetwork);
    if (ok) return true;
    return false;
}

// Check if the network is cyclic.
// Note the following lemma: "If D is a directed acyclic graph then D has at
// least one source (vertex of indegree 0) and at least one sink (vertex of
// outdegree 0)." Source: https://www.math.cmu.edu/~af1p/Teaching/GT/CH10.pdf
// I.e.: Add to the queue all nodes with indegree = 0. If there is no such node,
// then the network is cyclic. Otherwise, at each step, extract node X from the
// queue and for each node Y, with an edge E from from X to Y remove the edge E
// from the graph. If Y has no incoming edges (indegree(Y) = 0), add it to the
// queue and repeat this until the queue is empty. After that, if the graph
// still has edges, then it has at least one cycle. Otherwise, it is not cyclic.
// Note that I don`t actually remove edges from the graph, but I just subtract
// one from inDegree of a node, which means basically the same thing.
bool cyclic(graph *network) {
    initializeVisited(network);
    int *inDegree = initializeInDegree(network);
    int numNodes = countNodes(network);
    int queue[numNodes];
    int front = 0, back = 0, counter = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i) && inDegree[i] == 0) queue[back++] = i;
    }
    while (front < back) {
        int currentNode = queue[front++];
        node *neighbour = network->neighbours[currentNode];
        while (neighbour) {
            if (--inDegree[neighbour->key] == 0) queue[back++] = neighbour->key;
            neighbour = neighbour->next;
        }
        counter++;
    }
    free(inDegree);
    if (counter != numNodes) return true;
    return false;
}

// Return an array which represents topological ordering of an acyclic network.
// I.e.: Is actually the same algorithm for checking if a network is cyclic or
// not, but this time it returns an array of nodes sorted topological.
int *topologicalSort(graph *network) {
    initializeVisited(network);
    int *inDegree = initializeInDegree(network);
    int numNodes = countNodes(network);
    int *result = malloc(numNodes * sizeof(int));
    int queue[numNodes];
    int front = 0, back = 0, counter = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i) && inDegree[i] == 0) queue[back++] = i;
    }
    while (front < back) {
        int currentNode = queue[front++];
        node *neighbour = network->neighbours[currentNode];
        while (neighbour) {
            if (--inDegree[neighbour->key] == 0) queue[back++] = neighbour->key;
            neighbour = neighbour->next;
        }
        counter++;
    }
    for (int i = 0; i < back; i++) result[i] = queue[i];
    free(inDegree);
    return result;
}

// Detect a Hamiltonian cycle in the network using a "special" depth first
// search traversal.
// Why is it "special"? Because whenever it gets "stuck" (encounters a node that
// was already visited and the path is not yet hamiltonian), it marks the
// visited nodes from the recursive calls stack as not visited, so that it can
// generate all possible chains until it finds a closed chain of length equal to
// number of nodes + 1 (which means it is hamiltonian). In case it exhausted
// all possible chains, then no hamiltonian cycle was found.
bool hamiltonianCycle(graph *network, int currentNode, int startNode, int pathLength, int numNodes) {
    network->visited[currentNode] = true;
    // If startNode is amongst currentNode`s neighbours and the length of the
    // path is equal to the number of nodes in the network, then we found a
    // Hamiltonian cycle and return true.
    if (isNeighbour(network, currentNode, startNode) && pathLength == numNodes) return true;
    node *neighbour = network->neighbours[currentNode];
    while (neighbour) {
        if (! visited(network, neighbour->key)) {
            if (hamiltonianCycle(network, neighbour->key, startNode, pathLength + 1, numNodes)) return true;
            else network->visited[neighbour->key] = false;
        }
        neighbour = neighbour->next;
    }
    return false;
}

// Check if the network is Hamiltonian.
// I.e.: Do a "special" depth first search traversal, starting from an arbitrary
// node and return true if it detects a hamiltonian cycle. (See above)
bool hamiltonian(graph *network) {
    initializeVisited(network);
    int numNodes = countNodes(network);
    int startNode = -1;
    for (int i = 0; i < network->numNodes && startNode == -1; i++) {
        if (hasNode(network, i)) startNode = i;
    }
    return hamiltonianCycle(network, startNode, startNode, 1, numNodes);
}

// Check if the network is Eulerian.
// I.e.: A network is eulerian iff it is strongly connected and indegree and
// outdegree of every node are equal.
bool eulerian(graph *network) {
    initializeVisited(network);
    bool ok = true;
    if (stronglyConnected(network)) {
        int *inDegree = initializeInDegree(network);
        int *outDegree = initializeOutDegree(network);
        for (int i = 0; i < network->numNodes && ok; i++) {
            if (hasNode(network,i) && inDegree[i] != outDegree[i]) ok = false;
        }
        free(inDegree);
        free(outDegree);
    }
    else ok = false;
    if (ok) return true;
    return false;
}

// Check if a network is a directed tree.
// I.e.: Check if the network has exactly one node with indegree = 0 and no
// self-linked nodes. If it is not the case, then the network is not a tree.
// Otherwise, do a depth first search traversal starting from that node (root).
// After traversal, if there are remaining unvisited nodes, then the network is
// not a tree. Otherwise, the network is a tree.
bool isTree(graph *network) {
    initializeVisited(network);
    int *inDegree = initializeInDegree(network);
    int rootNodes = 0, root = -1;
    bool ok = true;
    for (int i = 0; i < network->numNodes && ok; i++) {
        if (hasNode(network, i)) {
            if (inDegree[i] == 0) {
                rootNodes++;
                root = i;
            }
            else if (linkedToItself(network, i)) ok = false;
        }
    }
    if (ok && rootNodes == 1) {
        DFS(network, root);
        if (! visitedAll(network)) ok = false;
    }
    else ok = false;
    free(inDegree);
    if (ok) return true;
    return false;
}

// Check if a given network (that is a directed tree) is also a binary tree.
// I.e.: Check if the tree has nodes with at most 2 children.
bool isBinaryTree(graph *network) {
    int *outDegree = initializeOutDegree(network);
    bool ok = true;
    for (int i = 0; i < network->numNodes && ok; i++) {
        if (hasNode(network, i) && outDegree[i] > 2) ok = false;
    }
    free(outDegree);
    if (ok) return true;
    return false;
}

// Return the root of a network, given the fact that the network is a
// directed tree.
int getRoot(graph *network) {
    int *inDegree = initializeInDegree(network);
    int root = -1;
    for (int i = 0; i < network->numNodes && root == -1; i++) {
        if (hasNode(network, i) && inDegree[i] == 0) root = i;
    }
    free(inDegree);
    return root;
}

// Return the depth of the network, given the fact that the network is a
// directed tree.
// Note that the depth of the root is 0.
int getDepth(graph *network, int root) {
    if (! hasNeighbours(network, root)) return 0;
    node *neighbour = network->neighbours[root];
    int depth = 0;
    while(neighbour) {
        depth = max(depth, getDepth(network, neighbour->key));
        neighbour = neighbour->next;
    }
    return 1 + depth;
}

// Do a depth first search traversal starting from the root of the tree and keep
// track of the depth of every node and its parent, storing this information in
// two arrays: depth and parent.
void preprocessLCA(graph *network, int prevNode, int currNode, int *depth, int *parent, int currDepth) {
    network->visited[currNode] = true;
    depth[currNode] = currDepth;
    parent[currNode] = prevNode;
    node *neighbour = network->neighbours[currNode];
    while (neighbour) {
        if (! visited(network, neighbour->key)) {
            preprocessLCA(network, currNode, neighbour->key, depth, parent, currDepth + 1);
        }
        neighbour = neighbour->next;
    }
}

// Return the lowest common ancestor of two given nodes.
// I.e.: Making use of the depth of every node and its parent, we go "up" in the
// tree with both nodes until we reach the same node.
int LCA(graph *network, int node1, int node2) {
    initializeVisited(network);
    int depth[network->numNodes], parent[network->numNodes];
    int root = getRoot(network);
    preprocessLCA(network, 0, root, depth, parent, 1);
    while (node1 != node2) {
        if (depth[node1] > depth[node2]) node1 = parent[node1];
        else node2 = parent[node2];
    }
    return node1;
}

// Extract an unvisited node with the minimum distance value.
int extractMin(graph *network, double *distance) {
    double min = DBL_MAX;
    int minNode = 0;
    for (int j = 0; j < network->numNodes; j++) {
        if (hasNode(network, j)) {
            if (distance[j] < min && ! visited(network, j)) {
                min = distance[j];
                minNode = j;
            }
        }
    }
    return minNode;
}

// Return the shortest distance from source to destination. If the destination
// node is not reachable, the result will be DBL_MAX.
double Dijkstra(graph *network, int nodeSource, int nodeDestination) {
    initializeVisited(network);
    double *dist = initializeDistance(network);
    dist[nodeSource] = 0;
    for (int i = 0; i < network->numNodes; i++) {
        if (hasNode(network, i)) {
            int minNode = extractMin(network, dist);
            network->visited[minNode] = true;
            node *neighbour = network->neighbours[minNode];
            while (neighbour) {
                if (dist[neighbour->key] > dist[minNode] + neighbour->weight) {
                    dist[neighbour->key] = dist[minNode] + neighbour->weight;
                }
                neighbour = neighbour->next;
            }
        }
    }
    double result = dist[nodeDestination];
    free(dist);
    return result;
}

// Check if the set of k nodes "subnetworkNodes" is a clique.
// I.e.: Check if the number of edges between that set of k nodes is equal to
// k * (k - 1).
bool clique(graph *network, int *subnetworkNodes, int k) {
    int counter = 0;
    for (int i = 1; i <= k; i++) {
        for (int j = 1; j <= k; j++) {
            if (isNeighbour(network, subnetworkNodes[i], subnetworkNodes[j]) &&
                subnetworkNodes[i] != subnetworkNodes[j]) {
                counter++;
            }
        }
    }
    if (counter == k * (k - 1)) return true;
    return false;
}

// Generate all the possible combinations of k nodes from a total of n and
// return the number of k-cliques through the "cliquesCounter" variable, which
// is passed by reference.
void kCliques(graph *network, int i, int n, int k, int *kNodes, int *nodes, int *cliquesCounter) {
    if (i == k + 1) {
        int subnetworkNodes[k + 1];
        for (int j = 1; j <= k; j++) subnetworkNodes[j] = nodes[kNodes[j]];
        if (clique(network, subnetworkNodes, k)) (*cliquesCounter)++;
    }
    else {
        for (int j = kNodes[i - 1] + 1; j <= n - k + i; j++) {
            kNodes[i] = j;
            kCliques(network, i + 1, n, k, kNodes, nodes, cliquesCounter);
        }
    }
}

// Return the number of k-cliques.
int countKCliques(graph *network, int k) {
    int numNodes = countNodes(network);
    int nodes[numNodes + 1], nodesCounter = 0;
    int kNodes[k + 1]; kNodes[0] = 0;
    for (int i = 0; i < network->numNodes; i++) {
          if (hasNode(network, i)) nodes[++nodesCounter] = i;
    }
    int cliquesCounter = 0;
    kCliques(network, 1, numNodes, k, kNodes, nodes, &cliquesCounter);
    return cliquesCounter;
}

// Return the number of cliques in the network.
int countCliques(graph *network) {
    int result = 0;
    int numNodes = countNodes(network);
    for (int k = 2; k <= numNodes; k++) {
        result += countKCliques(network, k);
    }
    return result;
}

// Extract information from the network.
void printInformation(graph *network, int node1, int node2) {
    printf("(1) Representation of the network: \n"); printNetwork(network);
    printf("(2) Number of nodes: %d\n", countNodes(network));
    printf("(3) Number of edges: %d\n", countEdges(network));
    printf("(4) Number of cliques: %d\n", countCliques(network));
    printf("(5) Is the network strongly connected? ");
    if (stronglyConnected(network)) printf("YES!\n"); else printf("NO!\n");
    printf("(6) Is the network cyclic? ");
    if (cyclic(network)) printf("YES!\n");
    else { printf("NO!\n");
        int *result = topologicalSort(network);
        printf("    (6.1) Topological sorting of the network: ");
        for (int i = 0; i < countNodes(network); i++) printf("%d ", result[i]);
        printf("\n");
        free(result);
    }
    printf("(7) Is the network Hamiltonian? ");
    if (hamiltonian(network)) printf("YES!\n"); else printf("NO!\n");
    printf("(8) Is the network Eulerian? ");
    if (eulerian(network)) printf("YES!\n"); else printf("NO!\n");
    printf("(9) Is the network a directed tree? ");
    if (isTree(network)) { printf("YES!\n");
        printf("    (9.1) Is it also a binary tree? ");
        if (isBinaryTree(network)) printf("YES!\n"); else printf("NO!\n");
        printf("    (9.2) Root: %d\n", getRoot(network));
        printf("    (9.3) Depth: %d\n", getDepth(network, getRoot(network)));
        if (node1 != -1 && node2 != -1) {
            printf("    (9.4) Lowest common ancestor of %d and %d: ", node1, node2);
            printf("%d\n", LCA(network, node1, node2));
        }
    }
    else printf("NO!\n");
    if (node1 != -1 && node2 != -1) {
        printf("(10) The shortest distance from %d to %d: ", node1, node2);
        double result = Dijkstra(network, node1, node2);
        if (result == DBL_MAX) printf("Node %d can`t be reached!\n", node2);
        else printf("%lf\n", result);
    }
}

//------------------------------------------------------------------------------
// ############################### PARSING DATA ################################
// -----------------------------------------------------------------------------

// Check if an isolated node can be added to network. If it is already connected
// in the network, return true and the node is not added. Otherwise, return
// false and the node is added.
bool existsConnected(graph *network, int isolatedNode) {
    if(network->neighbours[isolatedNode] != NULL) return true;
    for (int i = 0; i < network->numNodes; i++) {
        node* neighbour = network->neighbours[i];
        while (neighbour) {
            if(neighbour->key == isolatedNode) {
                return true;
            }
            neighbour = neighbour->next;
        }
    }
    return false;
}

// Check if a node exists already in the network as an isolated node, so that
// no edge "x-node" or "node-x" can be added to the network, where "node" is an
// isolated node in the network.
bool existsIsolated(graph *network, int node) {
    return (hasNode(network, node) && isolated(network, node));
}

// Check if an edge exists already in the network.
bool existsEdge(graph *network, int from, int to) {
    if (hasNode(network, from)) {
        node *neighbour = network->neighbours[from];
        while (neighbour) {
            if (neighbour->key == to) return true;
            neighbour = neighbour->next;
        }
    }
    return false;
}

// Convert a string into an integer. Return -1 if it is not valid.
int convertNode(const char node[]) {
    int n = atoi(node);
    char auxNode[MAX_DIGITS_INT];
    sprintf(auxNode, "%d", n);
    if (n >= 0 && n <= INT_MAX){
        if (strcmp(auxNode, node) == 0) return n;
    }
    return -1;
}

// Convert a string into a double. Return -1 if it is not valid.
double convertWeight(const char weight[]) {
    double n = strtod(weight, NULL);
    char auxWeight[MAX_DIGITS_DOUBLE];
    sprintf(auxWeight, "%f", n);
    if (n >= 0 && n <= DBL_MAX) {
        if(strncmp(auxWeight, weight, strlen(weight)) == 0) return n;
    }
    return -1;
}

// Convert the nodes from string into int and return them through the parameters
// "from" and "to", which are passed by reference.
// E.g.: Given the token "1-2/3.14", "from" will have the value 1 and "to" will
// have the value 2.
void getNodes(const char *tok, int *from, int *to) {
    int i, j;
    char q1[MAX_DIGITS_INT], q2[MAX_DIGITS_INT];
    for (i = 0; tok[i] != '-'; i++) q1[i] = tok[i];
    q1[i] = '\0';
    for (i = i + 1, j = 0; tok[i] != '/' && tok[i]; i++, j++) q2[j] = tok[i];
    q2[j] = '\0';
    *from = convertNode(q1);
    *to = convertNode(q2);
}

// Convert the weight from string to double and return it through the parameter
// "weight", which is passed by reference.
// E.g.: Given the token "1-2/3.14", "weight" will have the value 3.14.
void getWeight(const char *tok, double *weight) {
    char *p = strchr(tok, '/');
    if (p) {
        if (p[1]) *weight = convertWeight(p + 1);
        else *weight = -1;
    }
    else *weight = 1;
}

// Convert a string into a network. Return an empty network if the given string
// is invalid.
// I.e.: Separate the string into tokens like "x-y" or "x", where "x-y"
// represents a directed edge from x to y and "x" represents an isolated node x.
// Then, check if the converted tokens are valid and can be added to the network.
graph *convertData(const char args[]) {
    graph *network = newGraph();
    bool validData = true;
    int numNodes = 0;
    char auxArgs[strlen(args) + 1];
    strcpy(auxArgs, args);
    char *tok = strtok(auxArgs, ",");
    if (! isdigit(args[0])) validData = false;
    while(tok && validData) {
        char *q = strchr(tok, '-');
        if (q) {
            int from, to;
            double weight;
            getNodes(tok, &from, &to);
            getWeight(tok, &weight);
            if (from == -1 || to == -1 || weight == -1) validData = false;
            else {
                numNodes = max(numNodes, max(from,to));
                if (numNodes >= network->numNodes) expandNetwork(network, numNodes + 1);
                if (existsEdge(network, from, to)) validData = false;
                else if (existsIsolated(network, from) || existsIsolated(network, to)) validData = false;
                else addEdge(network, from, to, weight);
            }
        }
        else {
            int isolatedNode = convertNode(tok);
            if (isolatedNode == -1) validData = false;
            else {
                numNodes = max(numNodes, isolatedNode);
                if (numNodes >= network->numNodes) expandNetwork(network, numNodes + 1);
                if (existsConnected(network, isolatedNode)) validData = false;
                else addEdge(network, isolatedNode, -1, 0);
            }
        }
        tok = strtok(NULL, ",");
    }
    if (validData) return network;
    else {
        destroyNetwork(network);
        return newGraph();
    }
}

// -----------------------------------------------------------------------------
// ######################## USER INTERFACE AND TESTING #########################
// -----------------------------------------------------------------------------

// Use constants to say which function to call.
enum { Convert, CountNodes, CountEdges, CountCliques, StronglyConnected,
       Hamiltonian, Eulerian, Cyclic, Tree, BinaryTree, Root, Depth };
typedef int function;

// A replacement for the library assert function.
void assert(int line, bool b) {
    if (b) return;
    printf("The test on line %d fails.\n", line);
    exit(1);
}

// Call a given function, which is applied to the network, returning a possible
// integer or boolean result (or -1).
int call(function f, graph *network) {
    int result = -1;
    switch (f) {
      case Convert: result = ! emptyNetwork(network); break;
      case CountNodes: result = countNodes(network); break;
      case CountEdges: result = countEdges(network); break;
      case CountCliques: result = countCliques(network); break;
      case StronglyConnected: result = stronglyConnected(network); break;
      case Hamiltonian: result = hamiltonian(network); break;
      case Eulerian: result = eulerian(network); break;
      case Cyclic: result = cyclic(network); break;
      case Tree: result = isTree(network); break;
      case BinaryTree: result = isBinaryTree(network); break;
      case Root: result = getRoot(network); break;
      case Depth: result = getDepth(network, getRoot(network)); break;
      default: assert(__LINE__, false);
    }
    return result;
}

// Check that a given function does the right thing. The 'in' value is the
// network. The 'out' value is the expected result, or -1.
bool check(function f, char *in, int out) {
    graph *network = convertData(in);
    int result = call(f, network);
    bool ok = (result == out);
    destroyNetwork(network);
    return ok;
}

// Check if two arrays of integers are equal. Two arrays are considered to be
// equal iff both arrays contain the same number of elements, and all the
// corresponding pairs of elements in the two arrays are equal.
bool match(const int *arr1, const int *arr2, const int n) {
    for (int i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) return false;
    }
    return true;
}

// Check convertData().
void testConvertData() {
    assert(__LINE__, check(Convert, "0", true));
    assert(__LINE__, check(Convert, "1,2,3", true));
    assert(__LINE__, check(Convert, "1-2,3,2-2", true));
    assert(__LINE__, check(Convert, "1-2/3.14,3,2-2", true));
    assert(__LINE__, check(Convert, "1-2,-3", false));
    assert(__LINE__, check(Convert, "1-2,1", false));
    assert(__LINE__, check(Convert, "1,1-2", false));
    assert(__LINE__, check(Convert, "1,1", false));
    assert(__LINE__, check(Convert, "1-2,1-2", false));
    assert(__LINE__, check(Convert, "1-2/3,3/2", false));
}

// Check countNodes().
void testCountNodes() {
    assert(__LINE__, check(CountNodes, "1", 1));
    assert(__LINE__, check(CountNodes, "1,2,3", 3));
    assert(__LINE__, check(CountNodes, "1-2,3,2-2", 3));
    assert(__LINE__, check(CountNodes, "1-2,1000000", 3));
}

// Check countEdges().
void testCountEdges() {
    assert(__LINE__, check(CountEdges, "1", 0));
    assert(__LINE__, check(CountEdges, "1,2,3", 0));
    assert(__LINE__, check(CountEdges, "1-2,3,2-2", 2));
    assert(__LINE__, check(CountEdges, "1-2,1000000", 1));
}

// Check countCliques().
void testCountCliques() {
    assert(__LINE__, check(CountCliques, "1-2", 0));
    assert(__LINE__, check(CountCliques, "1-1", 0));
    assert(__LINE__, check(CountCliques, "1-2,2-1", 1));
    assert(__LINE__, check(CountCliques, "1-2,2-1,2-2", 1));
    assert(__LINE__, check(CountCliques, "1-2,3,2-2,4-5,5-4", 1));
    assert(__LINE__, check(CountCliques, "1-2,1-3,2-1,2-3,3-1,3-2", 4));
    assert(__LINE__, check(CountCliques, "1-2,1-3,2-1,2-3,3-1", 2));
    assert(__LINE__, check(CountCliques, "1-2,1-3,1-4,2-1,2-3,2-4,3-1,3-2,3-4,4-1,4-2,4-3", 11));
    assert(__LINE__, check(CountCliques, "1-2,1-3,1-4,2-1,2-3,2-4,3-1,3-2,3-4,4-1,4-2", 7));
    assert(__LINE__, check(CountCliques, "1-2,1-3,1-4,1-5,2-1,2-3,2-4,2-5,3-1,3-2,3-4,3-5,4-1,4-2,4-3,4-5,5-1,5-2,5-3,5-4", 26));
}

// Check stronglyConnected().
void testStronglyConnected() {
    assert(__LINE__, check(StronglyConnected, "0", true));
    assert(__LINE__, check(StronglyConnected, "1-1", true));
    assert(__LINE__, check(StronglyConnected, "1-2,2-3,3-1", true));
    assert(__LINE__, check(StronglyConnected, "1-2,2-3,3-1,2-4,4-2", true));
    assert(__LINE__, check(StronglyConnected, "1,2", false));
    assert(__LINE__, check(StronglyConnected, "1-2", false));
    assert(__LINE__, check(StronglyConnected, "1-2,3-2", false));
    assert(__LINE__, check(StronglyConnected, "1-2,2-3,1-3", false));
    assert(__LINE__, check(StronglyConnected, "1-2,2-3,3-1,4", false));
}

// Check cyclic().
void testCyclic() {
    assert(__LINE__, check(Cyclic, "1-1", true));
    assert(__LINE__, check(Cyclic, "1-2,2-3,3-1", true));
    assert(__LINE__, check(Cyclic, "1-2,2-3,3-4,4-4", true));
    assert(__LINE__, check(Cyclic, "1-2,3-4,4-5,5-3", true));
    assert(__LINE__, check(Cyclic, "1", false));
    assert(__LINE__, check(Cyclic, "1-2,2-3,4-1,4-3", false));
}

// check topologicalSort().
void testTopologicalSort() {
    graph *network = convertData("1-2,1-3,2-3,4-3,4-5,5-1");
    int *result = topologicalSort(network), test[] = {4, 5, 1, 2, 3};
    assert(__LINE__, match(result, test, 5) == true);
    destroyNetwork(network); free(result);
}

// Check hamiltonian().
void testHamiltonian() {
    assert(__LINE__, check(Hamiltonian, "1-2,2-3,3-4,4-1", true));
    assert(__LINE__, check(Hamiltonian, "1-2,2-3,3-4,4-5,5-1,4-2", true));
    assert(__LINE__, check(Hamiltonian, "1-2,2-3,3-4,4-5,5-1,2-4", true));
    assert(__LINE__, check(Hamiltonian, "1-2,2-1", true));
    assert(__LINE__, check(Hamiltonian, "1-1", true));
    assert(__LINE__, check(Hamiltonian, "1", false));
    assert(__LINE__, check(Hamiltonian, "1-2", false));
    assert(__LINE__, check(Hamiltonian, "1-2,2-3,3-1,4", false));
    assert(__LINE__, check(Hamiltonian, "1-2,2-3,3-1,4-3,3-4", false));
}

// Check eulerian().
void testEulerian() {
    assert(__LINE__, check(Eulerian, "1", true));
    assert(__LINE__, check(Eulerian, "1-2,2-3,3-4,4-5,5-1", true));
    assert(__LINE__, check(Eulerian, "1-2,2-3,3-1,4-3,3-4", true));
    assert(__LINE__, check(Eulerian, "1-2,2-3,3-1,4-3", false));
    assert(__LINE__, check(Eulerian, "1-2,2-3,3-4,4-5,5-1,4-2", false));
    assert(__LINE__, check(Eulerian, "1-2,2-3,3-1,4", false));
}

// Check isTree().
void testIsTree() {
    assert(__LINE__, check(Tree, "1", true));
    assert(__LINE__, check(Tree, "1-2,1-3,1-4,5-1", true));
    assert(__LINE__, check(Tree, "1-2,1-3,1-4,5-2", false));
    assert(__LINE__, check(Tree, "1-2,2-3,3-1,4-5", false));
    assert(__LINE__, check(Tree, "1-2,2-3,3-4,4-4", false));
    assert(__LINE__, check(Tree, "1-2,3-3", false));
    assert(__LINE__, check(Tree, "1-2,3", false));
}

// Check isBinaryTree().d
void testIsBinaryTree() {
    assert(__LINE__, check(BinaryTree, "1", true));
    assert(__LINE__, check(BinaryTree, "1-2,1-3,2-4,4-5,4-6", true));
    assert(__LINE__, check(BinaryTree, "1-2,1-3,2-4,4-5,4-6,4-7", false));
}

// Check getRoot().
void testGetRoot() {
    assert(__LINE__, check(Root, "1", 1));
    assert(__LINE__, check(Root, "1-2,1-3,1-4,5-1", 5));
    assert(__LINE__, check(Root, "1-2,1-3,2-4,4-5,4-6", 1));
}

// Check getDepth().
void testGetDepth() {
    assert(__LINE__, check(Depth, "1", 0));
    assert(__LINE__, check(Depth, "1-2,1-3,1-4", 1));
    assert(__LINE__, check(Depth, "1-2,2-3,2-4,3-5,3-6,1-7,7-8", 3));
}

// Check LCA().
void testLowestCommonAncestor() {
    graph *network = convertData("1-2,1-3,2-4,2-5,2-6,3-7");
    assert(__LINE__, LCA(network, 2, 2) == 2);
    assert(__LINE__, LCA(network, 4, 3) == 1);
    assert(__LINE__, LCA(network, 2, 7) == 1);
    assert(__LINE__, LCA(network, 1, 6) == 1);
    destroyNetwork(network);
}

// Check dijkstra().
void testDijkstra() {
    graph *network = convertData("1-2,1-4/2,1-5/9,2-3/3,3-5/2,4-3/1");
    assert(__LINE__, Dijkstra(network, 1, 3) == 3);
    assert(__LINE__, Dijkstra(network, 1, 5) == 5);
    assert(__LINE__, Dijkstra(network, 4, 1) == DBL_MAX); //Not reachable.
    destroyNetwork(network);
}

// Run tests on functions.
void test() {
    testConvertData();
    testCountNodes();
    testCountEdges();
    testCountCliques();
    testStronglyConnected();
    testCyclic();
    testTopologicalSort();
    testHamiltonian();
    testEulerian();
    testIsTree();
    testIsBinaryTree();
    testGetRoot();
    testGetDepth();
    testLowestCommonAncestor();
    testDijkstra();
    printf("All tests pass.\n");
}

// Run the program or, if there are no arguments, test it.
int main(int n, char *args[n]) {
    if (n == 1) {
        test();
    }
    else if (n == 2) {
        graph *network = convertData(args[1]);
        if (! emptyNetwork(network)) {
          printInformation(network, -1, -1); destroyNetwork(network);
        }
        else {
            destroyNetwork(network);
            fprintf(stderr, "You have introduced wrong data!\n"); exit(1);
        }
    }
    else if (n == 4) {
        graph *network = convertData(args[1]);
        int node1 = convertNode(args[2]), node2 = convertNode(args[3]);
        if (! emptyNetwork(network) && node1 != -1 && node2 != -1) {
            printInformation(network, node1, node2); destroyNetwork(network);
        }
        else {
            destroyNetwork(network);
            fprintf(stderr, "You have introduced wrong data!\n"); exit(1);
        }
    }
    else {
        fprintf(stderr, "Use e.g.: ./network 1-2/3.14,3,2-2.\n");
        exit(1);
    }
}
