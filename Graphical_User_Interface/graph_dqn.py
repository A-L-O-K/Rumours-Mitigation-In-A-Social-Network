import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx as nx

class GraphDQN(nn.Module):
    """
    Graph DQN model with two GCNConv layers followed by a Linear layer for predicting Q-values.
    """
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

def convert_nx_to_torch_geo(nx_graph):
    """
    Converts a NetworkX graph to a PyTorch Geometric data object.
    Assumes each node has a 'feature' attribute. If not, assigns random features.
    """
    for node in nx_graph.nodes():
        if 'feature' not in nx_graph.nodes[node]:
            nx_graph.nodes[node]['feature'] = torch.rand(10)  # Default feature size

    data = from_networkx(nx_graph)
    data.x = torch.stack([data.feature[i] for i in range(len(data.feature))])
    return data

def get_most_influential_nodes(model, nx_graph, top_k=5):
    """
    Finds the top_k most influential nodes in the graph based on model's Q-values.
    """
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
