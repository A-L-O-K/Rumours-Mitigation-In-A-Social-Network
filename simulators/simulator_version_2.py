import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import random

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

def simulate_message_spread_with_delay(graph, blocked_nodes, k=3):
    visited_nodes = set(blocked_nodes)  # Start with blocked nodes as visited
    nodes_by_step = []

    # Initialize the spread from k random nodes that are not blocked
    start_nodes = [node for node in graph.nodes() if node not in blocked_nodes]
    initial_nodes = random.sample(start_nodes, k)
    
    queue = deque(initial_nodes)
    visited_nodes.update(initial_nodes)
    nodes_by_step.append(set(initial_nodes))

    while queue:
        next_step_nodes = set()
        for _ in range(len(queue)):
            node = queue.popleft()
            # Get the unvisited neighbors of the node that are not blocked
            neighbors = list(set(graph.neighbors(node)) - visited_nodes - set(blocked_nodes))
            
            if neighbors:
                # Pass the message to one unvisited neighbor
                chosen_neighbor = random.choice(neighbors)
                next_step_nodes.add(chosen_neighbor)
                queue.append(chosen_neighbor)
                visited_nodes.add(chosen_neighbor)
        
        if next_step_nodes:
            nodes_by_step.append(next_step_nodes)
    
    return nodes_by_step

def update(num, graph, pos, nodes_by_step, ax, node_colors, blocked_nodes):
    ax.clear()
    
    # Draw nodes with initial colors
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, ax=ax, node_size=500)
    
    # Highlight nodes reached in this step
    reached_nodes = set()
    for i in range(num + 1):
        reached_nodes.update(nodes_by_step[i])
    
    # Set color of reached nodes and blocked nodes
    reached_colors = [
        'red' if node in blocked_nodes else 'green' if node in reached_nodes else 'lightblue'
        for node in graph.nodes()
    ]
    nx.draw(graph, pos, node_color=reached_colors, with_labels=True, ax=ax, node_size=500)

def animate_spread(graph, blocked_nodes, k=3):
    nodes_by_step = simulate_message_spread_with_delay(graph, blocked_nodes, k)

    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size for better visibility
    pos = nx.spring_layout(graph, k=0.9)  # Increase the spacing between nodes
    node_colors = ['lightblue' if node not in blocked_nodes else 'red' for node in graph.nodes()]
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(nodes_by_step), fargs=(graph, pos, nodes_by_step, ax, node_colors, blocked_nodes),
        interval=100, repeat=False
    )
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Read the graph from a file (replace '12831.edges' with your file path)
    graph = read_graph('12831.edges')
    
    # List of nodes to block (modify as needed)
    blocked_nodes = [13927832, 7899982, 180505807, 22229621, 353101127]
    
    # Run the animation with blocked nodes and k initial spread nodes
    animate_spread(graph, blocked_nodes, k=3)
