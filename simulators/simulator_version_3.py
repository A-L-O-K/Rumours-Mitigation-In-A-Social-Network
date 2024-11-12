import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
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
    ax.clear()
    
    # Draw nodes that are yet to be reached
    nx.draw(graph, pos, node_color=node_colors, with_labels=True, ax=ax, node_size=500)
    
    # Highlight nodes reached in this step
    reached_nodes = set()
    for i in range(num + 1):
        reached_nodes.update(nodes_by_step[i])
    
    # Set color of reached nodes
    reached_colors = ['green' if node in reached_nodes else 'lightblue' for node in graph.nodes()]
    
    # Color blocked nodes red
    for i, node in enumerate(graph.nodes()):
        if node in blocked_nodes:
            reached_colors[i] = 'red'
    
    nx.draw(graph, pos, node_color=reached_colors, with_labels=True, ax=ax, node_size=500)

def animate_spread(graph, start_nodes, blocked_nodes):
    nodes_by_step = simulate_message_spread_with_delay(graph, start_nodes, blocked_nodes)

    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size for better visibility
    pos = nx.spring_layout(graph, k=0.9)  # Increase the spacing between nodes
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

if __name__ == "__main__":
    graph = read_graph('12831.edges')
    
    # Input number of starting nodes k
    k = 3
    start_nodes = random.sample(list(graph.nodes()), k)
    
    # Input list of blocked nodes
    blocked_nodes = [10044992, 18976476, 18974638, 245421629, 15725851, 91443374, 180505807, 23771687, 19507576, 75939851, 35357461, 155696919, 109186893, 606083, 353101127, 275570583, 6368672, 136865608, 458897186, 7821492, 15236339, 198410517, 130268655, 22588327, 360882965, 14904282, 17938005, 398874773, 20221130, 10587552, 12081222, 15543818, 25583917, 21196272, 7461782, 165964253, 98065221, 18317233, 1678471, 668423, 19483147, 21145764, 1186, 17561826, 9611352, 16215038, 1656891, 15911247, 14, 305888449, 148943, 8940282, 17729005, 487072890, 19094625, 7899982, 13927832, 883301, 14202817, 16461070, 1859981, 9411772, 16004268, 17408993, 24340234, 9767472, 15801446, 13648692, 15109716, 15639334, 14819149, 9863222, 460693601, 14471007, 8132642, 5264791, 30331404, 17798441, 16934483, 19101100, 15304923, 15162141, 14086492, 14313755, 81962011, 12007182, 16734561, 638323, 47015938, 79033767, 5827292, 83971810, 17384098, 1371101, 287713, 163449492, 104937383, 6253882, 18248772, 14087951]
    # blocked_nodes = [1656891, 98065221, 245421629, 33633793, 8132642, 22229621, 5436752, 275570583, 10044992, 180505807, 23771687, 360882965, 19507576, 458897186, 198410517, 271605782, 353101127, 14202817, 20554406, 9411772, 155696919, 109186893, 19208772, 18248772, 15725851]
    # blocked_nodes = [10044992, 12081222, 23771687, 5436752, 1859981, 16734561, 180505807, 75939851, 15725851, 15109716, 130268655, 12544312, 16353769, 18317233, 7821492, 22229621, 91443374, 248049056, 360882965, 487851005, 15911247, 62415556, 487072890, 14313755, 17245224]
    
    animate_spread(graph, start_nodes, blocked_nodes)
