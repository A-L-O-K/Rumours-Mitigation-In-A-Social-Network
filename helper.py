import networkx as nx
import matplotlib.pyplot as plt
import random


def read_graph(file_path):
    G = nx.Graph() 

    with open(file_path, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.split()) 
            G.add_edge(node1, node2)

    return G

def display_graph(G):
    plt.figure(figsize=(10, 10)) 
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10) 
    plt.show()


def assign_colors_to_nodes(graph, circle_file):
    node_colors = {}  
    circles = []  
    
    with open(circle_file, 'r') as file:
        for line in file:
            parts = line.strip().split() 
            nodes_in_circle = parts[1:] 
            circles.append(nodes_in_circle)

    num_circles = len(circles)
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / num_circles) for i in range(num_circles)] 
    

    for color_index, nodes_in_circle in enumerate(circles):
        for node in nodes_in_circle:
            node_colors[node] = colors[color_index] 

    for node in graph.nodes:
        graph.nodes[node]['color'] = node_colors.get(str(node), (0.5, 0.5, 0.5)) 
    
    return graph

def display_colored_graph(graph):
    node_colors = [graph.nodes[node].get('color', (0.5, 0.5, 0.5)) for node in graph.nodes] 
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph, k=0.5, seed=42)  
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
    plt.show()

def color_specific_nodes_red(graph, red_node_list):

    # Step 1: Make all nodes gray
    for node in graph.nodes:
        graph.nodes[node]['color'] = 'gray'

    # Step 2: Color the nodes in the red_node_names list as red
    for node_name in red_node_list:
        # Check if the node_name exists in the graph
        if node_name in graph.nodes:
            graph.nodes[node_name]['color'] = 'red'

    return graph