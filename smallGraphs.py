import networkx as nx
import matplotlib.pyplot as plt
import random
import os

# Create a directory to save the images
output_dir = "graphs_images"
os.makedirs(output_dir, exist_ok=True)

# Function to create a larger connected graph with slight variations
def create_connected_graph(num_nodes=100):
    G = nx.Graph()
    
    # Add nodes from 1 to num_nodes
    G.add_nodes_from(range(1, num_nodes + 1))
    
    # Choose main nodes in each cluster
    main_node_1 = 5
    main_node_2 = 15
    
    # Add edges for the main nodes with random connections within clusters
    cluster_1_nodes = random.sample(range(1, num_nodes // 2 + 1), k=random.randint(20, 30))
    cluster_2_nodes = random.sample(range(num_nodes // 2 + 1, num_nodes + 1), k=random.randint(20, 30))
    
    edges_1 = [(main_node_1, i) for i in cluster_1_nodes if i != main_node_1]
    edges_2 = [(main_node_2, i) for i in cluster_2_nodes if i != main_node_2]
    
    G.add_edges_from(edges_1 + edges_2)
    
    # Add random cross-connections between the two clusters
    cross_edges = [(random.choice(cluster_1_nodes), random.choice(cluster_2_nodes)) for _ in range(10)]
    G.add_edges_from(cross_edges)
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        # Find all disconnected components and connect them
        components = list(nx.connected_components(G))
        for i in range(1, len(components)):
            # Pick one node from each disconnected component and connect them
            node_from_comp1 = random.choice(list(components[i - 1]))
            node_from_comp2 = random.choice(list(components[i]))
            G.add_edge(node_from_comp1, node_from_comp2)
    
    return G

# Generate and save 10 varied connected graphs
for i in range(10):
    G = create_connected_graph(num_nodes=100)
    node_colors = ['blue' if node <= 50 else 'green' for node in G.nodes]
    pos = nx.spring_layout(G, seed=42)  # Fix layout for consistency
    
    # Plot the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=300, font_color='white', font_size=8)
    
    # Save the graph as an image
    file_path = os.path.join(output_dir, f"connected_graph_{i + 1}.png")
    plt.savefig(file_path)
    plt.close()  # Close the plot to avoid overlapping figures

print(f"Connected graphs saved in the '{output_dir}' directory.")
