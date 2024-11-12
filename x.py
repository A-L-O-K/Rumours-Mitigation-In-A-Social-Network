import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx
import matplotlib.pyplot as plt

# Define the GraphDQN model
class GraphDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphDQN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x 

# Function to convert NetworkX graph to torch_geometric data
def convert_nx_to_torch_geo(nx_graph):
    for node in nx_graph.nodes():
        if 'feature' not in nx_graph.nodes[node]:
            nx_graph.nodes[node]['feature'] = torch.rand(10) 
    data = from_networkx(nx_graph)
    data.x = torch.stack([data.feature[i] for i in range(len(data.feature))])
    return data

# Function to get most influential nodes
def get_most_influential_nodes(model, nx_graph, top_k=5):
    model.eval()
    graph_data = convert_nx_to_torch_geo(nx_graph)
    
    with torch.no_grad():
        node_features = graph_data.x
        edge_index = graph_data.edge_index
        q_values = model(node_features, edge_index)
    
    _, top_indices = torch.topk(q_values.squeeze(), k=top_k)
    node_mapping = list(nx_graph.nodes)
    top_nodes = [node_mapping[i] for i in top_indices.tolist()]
    return top_nodes

# Streamlit interface
st.title("Graph-Based Influential Node Finder")

# Create and initialize the model
input_dim = 10
hidden_dim = 32
output_dim = 1
model = GraphDQN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Add nodes and edges
st.sidebar.subheader("Add Nodes and Edges")
num_nodes = st.sidebar.number_input("Number of nodes", min_value=1, max_value=100, value=10, step=1)
add_edge = st.sidebar.button("Add Random Edges")

G = nx.Graph()
G.add_nodes_from(range(num_nodes))

if add_edge:
    for i in range(num_nodes - 1):
        G.add_edge(i, (i + 1) % num_nodes)  # Connect sequentially and in a circle

# Run model and visualize
st.subheader("Graph Visualization")
fig, ax = plt.subplots()
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, ax=ax)
st.pyplot(fig)

if st.button("Run Model"):
    top_nodes = get_most_influential_nodes(model, G, top_k=5)
    st.write("Top Influential Nodes:", top_nodes)

    # Highlight top nodes
    colors = ['red' if node in top_nodes else 'skyblue' for node in G.nodes()]
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, ax=ax)
    st.pyplot(fig)
