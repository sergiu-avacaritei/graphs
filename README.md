# graphs
## GENERAL INFO ##
	> The main purpose of the program is to extract information from a given network about its structure and its properties.

## RUN ##
	> To run this program: ./network <data> or ./network <data> <node1> <node2>
		where:
		(*) <data> is a string of weighted links between nodes (e.g.: "1-2/3.14,3,2-2").
		(*) <node1> and <node2> are nodes from the network.

	> Note: I strongly recommend running it with: ./network <data> <node1> <node2> to see all of its functionalities
	(e.g: 1-2/3.14,1-3,2-4,2-5 1 4).

## ABOUT THE PROGRAM ##
	> Convert the string <data> into a network, checking for input errors.
	> Extract information from the network:
		(1) Representation of the network.
		(2) Number of nodes.
		(3) Number of edges.
		(4) Number of cliques.
		(5) Check if the network is strongly connected.
		(6) Check if the network is cyclic.
			(6.1) If not cyclic, print the topological ordering of the network.
		(7) Check if the network is Hamiltonian.
		(8) Check if the network is Eulerian.
		(9) Check if the network is a directed tree.
			(9.1) If it is the case, check if the network is also a binary tree.
			(9.2) Print the root of the tree.
			(9.3) Print the depth of the tree.
			(9.4) Print the lowest common ancestor of <node1> and <node2>.
		(10) Print the shortest distance from <node1> to <node2>.

## ABOUT STRUCTURES ##
	> The structure of the node:

		typdef struct node{
		int key;           => stores the key of the node.
		double weight;     => stores the weight of the edge that goes into that node.
		struct node *next; => pointer to the next node.
		}node;

	> The structure of the graph:

		typedef struct graph{
		int numNodes;   => It does not actually store the number of nodes, but the maximum key of a node in the network so that it is not restricted by inputs that have nodes with consecutive keys (e.g.: from 1 to n) and allows inputs like "80-90,100-340,500". Using this approach, it is essential to keep track of which nodes are actually in the network and which nodes are not, because, for instance, nodes from 1 to 79 are not in the network, though we allocate memory for 500 nodes.
		bool *visited;  => Keeps track of the visited nodes in the network (set true if the node is visited, false otherwise).
		bool *isolated; => Keeps track of the isolated nodes in the network (set true if the node is isolated, false otherwise).
		bool *nodes;    => Keeps track of the existing nodes in the network (set true if the node exists, false otherwise).
		struct node **neighbours; => Array of linked lists (i.e.: Every node in the network has a linked list and each node in this linked list represents the neighbour of the current node)
		}graph;
