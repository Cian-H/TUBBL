import networkx as nx
import pickle
import torch
from io import BytesIO
import pyintergraph
import numpy as np
import pandas as pd
import ast
import argparse

pd.set_option('display.max_colwidth', 1000)

class MappedUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)
    
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location=self._map_location)
        else:
            return super().find_class(module, name)

def get_top_nodes(model_name, path):
    graph_type = 'direct-freq-mul'
    co_activation = nx.read_graphml(f'{path}/neuron_output_{model_name.split("/")[-1]}_{graph_type}.graphml')

    device= None
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device_1 = torch.device("cuda:0")
            device_2 = torch.device("cuda:1")
            print("CUDA is available, using GPUs.")
        else:
            device = torch.device("cuda:0")
            print("CUDA is available, using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, using CPU.")

    print("Loading activations...")
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_JB.pkl", "rb") as f:
                jb_activations = MappedUnpickler(f, map_location='cuda:0').load()
            with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_CV.pkl", "rb") as f:
                cv_activations = MappedUnpickler(f, map_location='cuda:1').load()
        else:
            with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_JB.pkl", "rb") as f:
                jb_activations = MappedUnpickler(f, map_location='cuda:0').load()
            with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_CV.pkl", "rb") as f:
                cv_activations = MappedUnpickler(f, map_location='cuda:0').load()
    else:
        with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_JB.pkl", "rb") as f:
                jb_activations = MappedUnpickler(f, map_location='cpu').load()
            
        with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_CV.pkl", "rb") as f:
                cv_activations = MappedUnpickler(f, map_location='cpu').load()

    activations={}
    activations.update(jb_activations)
    activations.update(cv_activations)

    data=[]

    for item in activations.keys():
        dt = []
        dt.append(item)
        dt.append(activations[item]['position'])
        dt.append(activations[item]['activation'].cpu())
        dt.append(torch.topk(activations[item]['activation'],k=100)[1].tolist()) #top 100 neuron indices
        dt.append(activations[item]['output'])
        data.append(dt)

    for i in data:
        dt=[]
        for k in i[3]:
            dt.append(co_activation.nodes[str(k)]['token'])
        i.append(dt)

    df_direct_freq_mul = pd.DataFrame(data, columns=['id','position', 'activation', 'top100_indices', 'output', 'top100_tokens']).drop(columns=['activation'])
    df_direct_freq_mul.drop(df_direct_freq_mul[df_direct_freq_mul.position ==''].index, inplace=True)

    def check_intersection(output, top_tokens):
        output_tokens = set(output.split())
        out_tks=[]
        for token in output_tokens:
            if len(token) <= 2:
                t = list(token)
                for char in t:
                    out_tks.append(char)
            else:
                out_tks.append(token)
        output_tokens = set(out_tks)
        top_tokens_list = list(top_tokens)
        list_cleaned = [ast.literal_eval(token) if token.startswith('[') and token.endswith(']') else token for token in top_tokens_list]
        acc=[]
        for i in list_cleaned:
            top_tokens_set = set([token.split()[0] if len(token.split())>0 else '' for token in i])
            inter = len(top_tokens_set.intersection(output_tokens))
            acc.append(inter)

        return np.array(acc).sum()/(len(top_tokens)*len(top_tokens_set))

    df_direct_freq_mul['token_presence'] = df_direct_freq_mul.apply(lambda row: check_intersection(row['output'], row['top100_tokens']), axis=1)


    def get_subgraph(graph, nodes):
        subgraph = graph.subgraph([str(i) for i in nodes])
        return subgraph

    df_direct_freq_mul['subgraph'] = df_direct_freq_mul.apply(lambda row: get_subgraph(co_activation, row['top100_indices']), axis=1)

    df_direct_freq_mul.to_pickle(f'{path}/df_direct_freq_mul_{model_name.split("/")[-1]}.pkl')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Getting the top k nodes/neurons from co-activation graph.")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name')
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    args = parser.parse_args()
    model_name = args.model_name
    path = args.path

    get_top_nodes(model_name, path)