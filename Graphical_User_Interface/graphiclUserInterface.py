import streamlit as st
import networkx as nx
import torch
import matplotlib.pyplot as plt
from graph_dqn import GraphDQN, get_most_influential_nodes, convert_nx_to_torch_geo

# Load the pre-trained model
model = GraphDQN(input_dim=10, hidden_dim=32, output_dim=1)
model.load_state_dict(torch.load("graph_dqn_model.pth"))
model.eval()

st.title("Graph DQN - Node Influence Prediction")

# Predefined test graphs
def get_test_graph(name):
    if name == "Karate Club":
        return nx.karate_club_graph()
    elif name == "Erdős-Rényi":
        return nx.erdos_renyi_graph(34, 0.05)
    elif name == "Star Graph":
        return nx.star_graph(20)
    elif name == "Path Graph":
        return nx.path_graph(20)

# Initialize test_graph in session state if it’s not already there
if "test_graph" not in st.session_state:
    st.session_state.test_graph = None

# Test Graph Selection Section
st.header("Select a Test Graph")
test_graph_options = ["Karate Club", "Erdős-Rényi", "Star Graph", "Path Graph"]
selected_test_graph = st.selectbox("Choose a test graph", test_graph_options)
top_k_test = st.slider("Select Top K Influential Nodes for Test Graph", min_value=1, max_value=10, value=5)

# Button to load the test graph and store it in session state
if st.button("Load Test Graph"):
    st.session_state.test_graph = get_test_graph(selected_test_graph)
    st.write(f"{selected_test_graph} loaded.")

# Show the graph if it has been loaded
if st.session_state.test_graph is not None:
    if st.button("Show Test Graph"):
        pos = nx.spring_layout(st.session_state.test_graph)
        color_map = ["blue" for _ in st.session_state.test_graph.nodes]
        
        plt.figure(figsize=(8, 6))
        nx.draw(st.session_state.test_graph, pos, node_color=color_map, with_labels=True, node_size=500, font_color="white")
        st.pyplot(plt)

    if st.button("Run Influential Node Detection on Test Graph"):
        top_nodes_test = get_most_influential_nodes(model, st.session_state.test_graph, top_k=top_k_test)
        st.write(f"Top {top_k_test} Influential Nodes:", top_nodes_test)

        pos = nx.spring_layout(st.session_state.test_graph)
        color_map = ["red" if node in top_nodes_test else "blue" for node in st.session_state.test_graph.nodes]
        
        plt.figure(figsize=(8, 6))
        nx.draw(st.session_state.test_graph, pos, node_color=color_map, with_labels=True, node_size=500, font_color="white")
        st.pyplot(plt)
else:
    st.warning("Please load a test graph first.")

# Custom Graph Creation Section
st.header("Custom Graph Creation")

node_count = st.slider("Number of Nodes for Custom Graph", min_value=5, max_value=50, value=20, step=1)
G = nx.Graph()
G.add_nodes_from(range(node_count))

if st.button("Add Random Edges"):
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if torch.rand(1).item() < 0.1:  # 10% chance of adding an edge
                G.add_edge(i, j)

# Custom Node Addition
new_node_id = st.number_input("Add Node ID to Custom Graph", min_value=0, step=1)
if st.button("Add Node to Custom Graph"):
    G.add_node(new_node_id)

node1 = st.number_input("Connect from Node (Custom Graph)", min_value=0, step=1)
node2 = st.number_input("Connect to Node (Custom Graph)", min_value=0, step=1)
if st.button("Add Edge to Custom Graph"):
    G.add_edge(node1, node2)

if st.button("Show Custom Graph"):
    pos = nx.spring_layout(G)
    color_map = ["blue" for _ in G.nodes]
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=500, font_color="white")
    st.pyplot(plt)

# Influence Prediction for Custom Graph
top_k_custom = st.slider("Select Top K Influential Nodes for Custom Graph", min_value=1, max_value=10, value=5)
if st.button("Run Influential Node Detection on Custom Graph"):
    top_nodes_custom = get_most_influential_nodes(model, G, top_k=top_k_custom)
    st.write(f"Top {top_k_custom} Influential Nodes:", top_nodes_custom)

    pos = nx.spring_layout(G)
    color_map = ["red" if node in top_nodes_custom else "blue" for node in G.nodes]
    
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, node_color=color_map, with_labels=True, node_size=500, font_color="white")
    st.pyplot(plt)
