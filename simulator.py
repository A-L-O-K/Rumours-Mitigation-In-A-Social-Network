import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

def simulate_message_spread_with_delay(graph, start_nodes):
    visited_nodes = set()
    nodes_by_step = []

    # Queue for BFS
    queue = deque(start_nodes)
    visited_nodes.update(start_nodes)
    nodes_by_step.append(set(start_nodes))

    while queue:
        next_step_nodes = set()
        # For each node in the current queue, add its neighbors to the next step
        for _ in range(len(queue)):
            node = queue.popleft()
            neighbors = set(graph.neighbors(node)) - visited_nodes  # Unvisited neighbors
            next_step_nodes.update(neighbors)
        
        if next_step_nodes:
            nodes_by_step.append(next_step_nodes)
            queue.extend(next_step_nodes)
            visited_nodes.update(next_step_nodes)
    
    return nodes_by_step

def update(num, graph, pos, nodes_by_step, ax, node_colors):
    ax.clear()
    
    # Draw nodes that are yet to be reached
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, ax=ax, node_size=500)
    
    # Highlight nodes reached in this step
    reached_nodes = set()
    for i in range(num + 1):
        reached_nodes.update(nodes_by_step[i])
    
    # Set color of reached nodes
    reached_colors = ['green' if node in reached_nodes else 'lightblue' for node in graph.nodes()]
    nx.draw(graph, pos, node_color=reached_colors, with_labels=True, ax=ax, node_size=500)

def animate_spread(graph, start_nodes):
    nodes_by_step = simulate_message_spread_with_delay(graph, start_nodes)

    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size for better visibility
    pos = nx.spring_layout(graph, k=0.9)  # Increase the spacing between nodes
    node_colors = ['lightblue'] * len(graph.nodes())
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(nodes_by_step), fargs=(graph, pos, nodes_by_step, ax, node_colors),
        interval=1000, repeat=False
    )
    
    plt.show()


def read_graph(file_path):
    G = nx.Graph()

    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    node1, node2 = map(int, line.split())
                    G.add_edge(node1, node2)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    
    return G

# ----------------------------------------------------------------------------------------------------------------------
# # Create a graph
# G = nx.Graph()

# # Add nodes 1 to 20
# G.add_nodes_from(range(1, 21))

# # Add edges for node 5
# edges_5 = [(5, i) for i in [1, 2, 3, 4, 6, 7, 8, 9, 10]]
# G.add_edges_from(edges_5)

# # Add edges for node 15
# edges_15 = [(15, i) for i in [11, 12, 13, 14, 16, 17, 18, 19, 20]]
# G.add_edges_from(edges_15)

# # Add some connections between the two sub-networks
# cross_edges = [(3, 12), (7, 18), (2, 15), (9, 19), (4, 11)]  # Sample cross-connections
# G.add_edges_from(cross_edges)

# G = "12831.edges"
# test_graph = read_graph(G)

if __name__ == "__main__":
    graph = read_graph('12831.edges')
    print(type(graph))  # Check if this outputs <class 'networkx.classes.graph.Graph'>
    # start_nodes = [13927832, 7899982, 180505807, 22229621, 353101127]
    start_nodes = [353101127]

    animate_spread(graph, start_nodes)