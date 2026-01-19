# %%
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from joblib import Parallel, delayed
from io import BytesIO
import argparse

class MappedUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)
    
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location=self._map_location)
        else:
            return super().find_class(module, name)

def inverse_function(model_name, path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_JB.pkl", "rb") as f:
            jb_activations = MappedUnpickler(f, map_location='cpu').load()
        
    with open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_CV.pkl", "rb") as f:
            cv_activations = MappedUnpickler(f, map_location='cpu').load()
 
    activations={}
    activations.update(jb_activations)
    activations.update(cv_activations)


    def to_tensor_safe(data):
        """Convert to tensor with copy to avoid shared memory issues"""
        if torch.is_tensor(data):
            return data.detach().clone()
        else:
            # torch.as_tensor handles numpy arrays and other types
            return torch.tensor(data).detach().clone()

    def return_token(hidden_states, decoded=True):
        if not None:
            #logits = model.lm_head(model.model.norm(torch.tensor(hidden_states)))
            logits = model.lm_head(model.model.norm(to_tensor_safe(hidden_states)))
            if decoded:
                return tokenizer.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
            else:
                return torch.argmax(logits, dim=-1).item()
        else:
            return 'None'

    new_activations = {}

    def process_token(key, token, activation_data):
        original_token = return_token(activation_data, False)
        return original_token, activation_data

    for key in tqdm(activations.keys()):
        new_activations[key] = {}
        new_activations[key]['position'] = activations[key]['position']
        new_activations[key]['output'] = activations[key]['output']
        new_activations[key]['activation'] = activations[key]['activation']
        new_activations[key]['sim_out'] = {}

        # Parallel execution with joblib
        results = Parallel(n_jobs=-1, verbose=0, prefer='threads')(
            delayed(process_token)(
                key, token, activations[key]['sim_out'][token]
            )
            for token in activations[key]['sim_out'].keys()
        )
        
        for original_token, hidden_features in results:
            new_activations[key]['sim_out'][
            original_token
            ] = hidden_features


    pickle.dump(new_activations, open(f"{path}/new_activations_direct_{model_name.split('/')[-1]}.pkl", "wb"))
    print("Done saving new activations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inverse hidden states to token script.")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name')
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    args = parser.parse_args()
    model_name = args.model_name
    path = args.path
    inverse_function(model_name, path)