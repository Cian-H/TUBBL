import networkx as nx
import pickle
import torch
from io import BytesIO
import numpy as np
import pandas as pd
import ast
from difflib import SequenceMatcher
import json
from tqdm import tqdm
from operator import itemgetter
import argparse

def get_subgraph(graph, nodes,k):
    subgraph_nodes_sorted = sorted(
        [(node, centrality[node]) for node in nodes if node in centrality],
        key=itemgetter(1),
        reverse=True
    )

    # Get just the nodes in order (highest to lowest centrality)
    ordered_nodes = [node for node, _ in subgraph_nodes_sorted]
    return ordered_nodes[:k]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Getting the top k nodes/neurons from co-activation graph.")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name')
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    parser.add_argument("--k", type=int, default=10, help="Number of top nodes to select.", required=False)
    args = parser.parse_args()
    model_name = args.model_name
    path = args.path
    k = args.k

    graph_type = 'direct-freq-mul'
    co_activation = nx.read_graphml(f'{path}neuron_output_{model_name.split("/")[-1]}_{graph_type}.graphml')

    df_all_nodes = pd.read_pickle(f'{path}df_direct_all_nodes_skills.pkl')

    centrality = nx.degree_centrality(co_activation)

    df_all_nodes['top10_nodes_skills'] = df_all_nodes.apply(lambda row: get_subgraph(co_activation, row['nodes'],k), axis=1)

    pickle.dump(df_all_nodes, open(f'{path}df_direct_all_top{k}_nodes_skills_{model_name.split("/")[-1]}.pkl', 'wb'))