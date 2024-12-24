import networkx as nx

# Create a directed graph for Out-Degree
G = nx.DiGraph()

# Add edges to the graph
edges = [("A", "B"), ("A", "D"), ("B", "C"), ("B", "E"), ("E", "F"), ("C", "F")]
G.add_edges_from(edges)

# 1. Calculate Out-Degree for all nodes
print("Out-Degree of each node:")
out_degree = {node: G.out_degree(node) for node in G.nodes()}
for node, degree in out_degree.items():
    print(f"Node {node}: {degree}")
x=list(out_degree.items())
x.sort(key=lambda x: x[1], reverse=True)
print("\nTop 3 nodes with highest out-degree:", x[:3])

# 2. Calculate Betweenness Centrality
print("\nBetweenness Centrality of each node:")
betweenness_centrality = nx.betweenness_centrality(G)

for node, centrality in betweenness_centrality.items():
    print(f"Node {node}: {centrality:.3f}")
y=list(betweenness_centrality.items())
y.sort(key=lambda x: x[1], reverse=True)
print("\nTop 3 nodes with highest betweenness centrality:", y[:3])
# Visualizing the graph (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Positions for nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=15)
plt.title("Graph Visualization")
plt.show()
