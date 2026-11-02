"""
graph_features.py - Graph Structural Feature Extraction for Fake Account Detection.

Project: "A Graph-and-Text Feature Classifier-Based Approach to Fake Social-Media Account Detection"
Academic Note:
    Social media follower/following relationships are modeled as a directed graph G = (V, E).
    IMPORTANT: The edge-list dataset used in this prototype (data/illustrative_account_edges.csv)
    is an illustrative prototype graph constructed specifically to demonstrate NetworkX graph
    feature extraction pipelines. It is NOT scraped from a live production social media platform.
"""

import os
import networkx as nx
import pandas as pd
import numpy as np


def build_account_graph(
    usernames: list,
    edge_list_path: str = os.path.join("data", "illustrative_account_edges.csv")
) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph (DiGraph) representing account interaction networks.
    
    Parameters:
        usernames (list): List of all account usernames from the primary dataset.
        edge_list_path (str): Filepath to the CSV containing 'source', 'target' follow relationships.
        
    Returns:
        nx.DiGraph: Directed graph with all accounts as nodes and relationships as directed edges.
    """
    G = nx.DiGraph()
    
    # Add all accounts as nodes first to ensure isolated nodes (degree = 0) are preserved
    for user in usernames:
        G.add_node(str(user))
        
    if not os.path.exists(edge_list_path):
        print(f"[Graph Features Warning] Edge-list file not found at: {edge_list_path}")
        print("  Proceeding with an empty graph where all nodes have 0 degree.")
        return G
        
    edges_df = pd.read_csv(edge_list_path)
    required_cols = {"source", "target"}
    if not required_cols.issubset(edges_df.columns):
        raise ValueError(f"Edge-list CSV must contain 'source' and 'target' columns. Found: {edges_df.columns}")
        
    # Add directed edges: source follows target (source -> target)
    for _, row in edges_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        G.add_edge(src, tgt)
        
    return G


def extract_graph_features(
    G: nx.DiGraph,
    usernames: list
) -> pd.DataFrame:
    """
    Calculates topological graph features for each account in the network.
    
    Features Extracted & Algorithmic Explanations:
    1. in_degree: Number of incoming directed edges (accounts following this user).
       - Higher in-degree is typical of popular, trusted, or authority profiles.
    2. out_degree: Number of outgoing directed edges (accounts this user follows).
       - Abnormally high out-degree often signals aggressive automated follow behavior in bots.
    3. total_degree: in_degree + out_degree.
    4. in_out_degree_ratio: in_degree / (out_degree + 1).
       - Ratio metric: values << 1 indicate bot-like asymmetric follow behavior.
    5. pagerank: Measures node prestige/influence based on incoming links from authoritative nodes.
    6. betweenness_centrality: Fraction of all shortest paths passing through a node.
       - Identifies bridge nodes linking disparate communities.
    7. clustering_coefficient: Fraction of possible triangles through a node that exist.
       - Evaluates local neighborhood density and organic community formation.
    8. weak_component_size: Size of the weakly connected component containing the node.
    9. strong_component_size: Size of the strongly connected component (mutually reachable ring).
       - Botnets often form closed directed strongly connected components.
       
    Parameters:
        G (nx.DiGraph): The directed social network graph.
        usernames (list): List of account handles to extract features for.
        
    Returns:
        pd.DataFrame: Feature matrix of graph metrics aligned with the input usernames.
    """
    features = []
    
    # Pre-calculate global network measures
    pagerank_dict = nx.pagerank(G, alpha=0.85) if len(G) > 0 else {}
    betweenness_dict = nx.betweenness_centrality(G) if len(G) > 0 else {}
    clustering_dict = nx.clustering(G.to_undirected()) if len(G) > 0 else {}
    
    # Pre-calculate component sizes
    weak_components = list(nx.weakly_connected_components(G))
    strong_components = list(nx.strongly_connected_components(G))
    
    # Map node to component size
    node_to_weak_size = {}
    for comp in weak_components:
        comp_size = len(comp)
        for node in comp:
            node_to_weak_size[node] = comp_size
            
    node_to_strong_size = {}
    for comp in strong_components:
        comp_size = len(comp)
        for node in comp:
            node_to_strong_size[node] = comp_size
            
    for user in usernames:
        user_str = str(user)
        if user_str in G:
            in_deg = float(G.in_degree(user_str))
            out_deg = float(G.out_degree(user_str))
            tot_deg = in_deg + out_deg
            in_out_ratio = in_deg / (out_deg + 1.0)
            pr = float(pagerank_dict.get(user_str, 0.0))
            bw = float(betweenness_dict.get(user_str, 0.0))
            cc = float(clustering_dict.get(user_str, 0.0))
            w_size = float(node_to_weak_size.get(user_str, 1.0))
            s_size = float(node_to_strong_size.get(user_str, 1.0))
        else:
            in_deg, out_deg, tot_deg, in_out_ratio = 0.0, 0.0, 0.0, 0.0
            pr, bw, cc, w_size, s_size = 0.0, 0.0, 0.0, 0.0, 0.0
            
        features.append({
            "username": user,
            "graph_in_degree": in_deg,
            "graph_out_degree": out_deg,
            "graph_total_degree": tot_deg,
            "graph_in_out_ratio": in_out_ratio,
            "graph_pagerank": pr,
            "graph_betweenness": bw,
            "graph_clustering_coefficient": cc,
            "graph_weak_component_size": w_size,
            "graph_strong_component_size": s_size
        })
        
    graph_df = pd.DataFrame(features)
    return graph_df


# =====================================================================
# Standalone Execution / Verification Block
# =====================================================================
if __name__ == "__main__":
    accounts_csv = os.path.join("data", "processed_accounts.csv")
    if not os.path.exists(accounts_csv):
        accounts_csv = os.path.join("data", "social_accounts.csv")
        
    edges_csv = os.path.join("data", "illustrative_account_edges.csv")
    
    print("=" * 65)
    print(" TASK 2: GRAPH FEATURE EXTRACTION (NETWORKX DIRECTED ANALYSIS)")
    print("=" * 65)
    print(f"Accounts Dataset: {accounts_csv}")
    print(f"Edge-list Dataset: {edges_csv}")
    
    df_acc = pd.read_csv(accounts_csv)
    usernames = list(df_acc["username"])
    
    # 1. Build NetworkX DiGraph
    G = build_account_graph(usernames, edge_list_path=edges_csv)
    print(f"\n[Graph Structure Summary]:")
    print(f"  Total Nodes (Accounts): {G.number_of_nodes()}")
    print(f"  Total Directed Edges  : {G.number_of_edges()}")
    print(f"  Graph Density         : {nx.density(G):.4f}")
    
    # 2. Extract Graph Features
    graph_features_df = extract_graph_features(G, usernames)
    
    print("\n[Extracted Graph Features for All Accounts]:")
    display_cols = [
        "username", "graph_in_degree", "graph_out_degree",
        "graph_in_out_ratio", "graph_pagerank", "graph_clustering_coefficient",
        "graph_strong_component_size"
    ]
    print(graph_features_df[display_cols].to_string(index=False))
    
    print("\n[Metric Justifications for Fake Account Detection]:")
    print("  - 'graph_in_out_ratio': Genuine nodes have higher incoming follows; bots have near-zero ratios.")
    print("  - 'graph_pagerank': Authority nodes (e.g. news_daily) score highest due to multiple incoming endorsements.")
    print("  - 'graph_strong_component_size': Mutual follow-rings among bots show up as strongly connected groups.")
    print("=" * 65)
    print("Graph feature extraction completed successfully!")
    print("=" * 65)
