import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from mpl_toolkits.mplot3d import Axes3D
import random

def simulate_message_spread_with_delay(graph, start_nodes, blocked_nodes):
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
            neighbors = set(graph.neighbors(node)) - visited_nodes - set(blocked_nodes)  # Exclude blocked nodes
            next_step_nodes.update(neighbors)
        
        if next_step_nodes:
            nodes_by_step.append(next_step_nodes)
            queue.extend(next_step_nodes)
            visited_nodes.update(next_step_nodes)
    
    return nodes_by_step

def update(num, graph, pos, nodes_by_step, ax, node_colors, blocked_nodes):
    ax.cla()  # Clear the plot for the new frame
    ax.set_axis_off()  # Turn off axis for clean view

    # Draw nodes that are yet to be reached
    for node in graph.nodes():
        x, y, z = pos[node]
        color = 'lightblue' if node not in blocked_nodes else 'red'
        ax.scatter(x, y, z, color=color, s=100)

    # Highlight nodes reached in this step
    reached_nodes = set()
    for i in range(num + 1):
        reached_nodes.update(nodes_by_step[i])

    # Update the colors for the reached nodes
    for node in reached_nodes:
        x, y, z = pos[node]
        ax.scatter(x, y, z, color='green', s=100)

    # Draw edges between nodes
    for edge in graph.edges():
        x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
        y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
        z_vals = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.5)

def animate_spread(graph, start_nodes, blocked_nodes):
    nodes_by_step = simulate_message_spread_with_delay(graph, start_nodes, blocked_nodes)

    fig = plt.figure(figsize=(10, 10))  # Adjust figure size
    ax = fig.add_subplot(111, projection='3d')
    pos = nx.spring_layout(graph, dim=3, k=0.9)  # Set up 3D layout positions with spacing

    node_colors = ['lightblue'] * len(graph.nodes())
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(nodes_by_step), fargs=(graph, pos, nodes_by_step, ax, node_colors, blocked_nodes),
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

if __name__ == "_main_":
    graph = read_graph('./12831.edges')
    
    # Input number of starting nodes k
    k = 3
    start_nodes = random.sample(list(graph.nodes()), k)
    
    # Input list of blocked nodes
    blocked_nodes = [10044992, 18976476, 18974638, 245421629, 15725851, 91443374, 180505807, 23771687, 19507576, 75939851, 35357461, 155696919, 109186893, 606083, 353101127, 275570583, 6368672, 136865608, 458897186, 7821492, 15236339, 198410517, 130268655, 22588327, 360882965, 14904282, 17938005, 398874773, 20221130, 10587552, 12081222, 15543818, 25583917, 21196272, 7461782, 165964253, 98065221, 18317233, 1678471, 668423, 19483147, 21145764, 1186, 17561826, 9611352, 16215038, 1656891, 15911247, 14, 305888449, 148943, 8940282, 17729005, 487072890, 19094625, 7899982, 13927832, 883301, 14202817, 16461070, 1859981, 9411772, 16004268, 17408993, 24340234, 9767472, 15801446, 13648692, 15109716, 15639334, 14819149, 9863222, 460693601, 14471007, 8132642, 5264791, 30331404, 17798441, 16934483, 19101100, 15304923, 15162141, 14086492, 14313755, 81962011, 12007182, 16734561, 638323, 47015938, 79033767, 5827292, 83971810, 17384098, 1371101, 287713, 163449492, 104937383, 6253882, 18248772, 14087951]
    
    animate_spread(graph, start_nodes, blocked_nodes)