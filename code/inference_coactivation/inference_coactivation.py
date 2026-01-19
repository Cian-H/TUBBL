
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from datasets import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import spacy
import re
from collections import defaultdict
import pickle
import multiprocessing
from multiprocessing import Pool, get_context
import argparse
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
import torch._dynamo
from io import BytesIO
import gc

torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 999999999

nlp = spacy.load("en_core_web_sm")  # Load spaCy model for NLP tasks
def extract_job_title(text):
    """
    Extracts the job title from the given text using spaCy.
    """
    doc = nlp(text)
    # Assuming the first noun phrase is the job title
    for t in doc.ents:
        if type(t.text) == str:
            text = text.replace(t.text, "")
    return re.sub(r'[^a-zA-Z ]+', ' ',text).lstrip().rstrip().replace('   ',' ').replace('  ',' ')

def create_skill_inference_prompt(text, role=None):
    """
    Creates a structured message list for skill inference from the given text.
    Returns messages in the format expected by apply_chat_template.
    
    Args:
        text: The job title to analyze
        role: Optional role specification (e.g., "Human Resources specialist")
              If None, defaults to "expert in the Human Resources department"
    
    Returns:
        List of message dictionaries with role and content keys
    """
    system_role = role if role else "expert in the Human Resources department"
    
    messages = [
        {
            "role": "system", 
            "content": f"You are an {system_role}. Analyze job title and infer up to 20 key skills related to them. Return ONLY the skills inferred, separated by commas, without any introduction or additional text."
        },
        {
            "role": "user", 
            "content": f"Job title: {text}"
        }
    ]
    
    return messages


# 1. Data Loader and Preprocessor
def load_and_preprocess_data(dataset_name, tokenizer, device, num_samples):
    dataset = load_dataset(dataset_name, split="train").select(range(num_samples))
    def tokenize_function(examples):
        if dataset_name == JOB_DESCRIPTION_DATASET:
            text_column = "Long Description"
        elif dataset_name == CV_DATASET:
            text_column = "CV"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        if text_column not in examples:
            raise ValueError(f"Dataset {dataset_name} does not have a \'{text_column}\' column.")
        examples['position_impr'] = []
        for pos in examples['Position']:
            examples['position_impr'].append(extract_job_title(pos))

        # Create message lists for each job title
        message_lists = [
            create_skill_inference_prompt(text) for text in examples['position_impr']
        ]

        # Apply chat template to format the conversations
        prompted_texts = []
        for messages in message_lists:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # Get text first, then tokenize
                add_generation_prompt=True
            )
            prompted_texts.append(formatted_text)

        # Store the original text alongside tokenized data
        tokenized_output = tokenizer(prompted_texts, truncation=True, padding="max_length", max_length=512).to(device)
        tokenized_output["text_original"] = examples[text_column]
        tokenized_output["position"] = examples["position_impr"]
        return tokenized_output
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "text_original", "position"]) # Add original text
    return torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)

# 2. Model Handler
def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def hook_function(model_name, model):
    # 3. Activation Extractor
    global activations, act_list
    activations={}
    act_list = []
    def get_activation(name):
        def hook(model, input, output):
            if type(output) != tuple:
                act_list.extend(output[:, -1:, :].detach())
            else:
                act_list.extend(output[0][:, -1:, :].detach())
            activations[name] = act_list
        return hook
# Register hook for the last MLP layer
# For GPT2, MLP layers are typically within model.transformer.h[i].mlp.c_proj
# For Llama3, it would be model.model.layers[15]
    last_mlp_layer = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for layer in reversed(model.transformer.h):
            if hasattr(layer, "mlp"): 
                last_mlp_layer = layer.mlp
                break
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in reversed(model.model.layers):       
            last_mlp_layer = layer
            break

    if last_mlp_layer:
        hook = last_mlp_layer.register_forward_hook(get_activation("last_mlp_layer"))
        print(f"Hook registered on: {last_mlp_layer}")
    else:
        print("Could not find the last MLP layer. Please inspect the model structure.")
        exit()

def find_most_similar_word(part_word, sentence):
    words = sentence.split()
    best_match = None
    best_ratio = 0
    
    for word in words:
        # Clean the word from punctuation
        clean_word = word.strip('.,!?;:"')
        similarity = SequenceMatcher(None, part_word.lower(), clean_word.lower()).ratio()
        
        if similarity > best_ratio:
            best_ratio = similarity
            best_match = clean_word
    
    return best_match, best_ratio

def get_activation_map(model_name, model, tokenizer, max_new_tokens, dataloader, dataloader_name, device, do_sample, top_k, top_p, temperature, path):
    all_mlp_activations_and_texts = {}

    print(f"Processing {dataloader_name}...")
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            similarity_input={}
            similarity_output={}
            # Generate text from the model
            generated_output = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_new_tokens, # Generate a short sequence
                do_sample=do_sample, # Use sampling for more diverse output
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id # Use eos_token_id for padding
            )

            output_tokens = generated_output[0][batch["input_ids"].shape[1]:]
 
            try: 
                for id, tk_out in enumerate(output_tokens):
                    if id == 0 or id > max_new_tokens - 1 or len(tokenizer.decode(tk_out,skip_special_tokens=True).strip()) == 0: #Skip the first token (usually the start token)
                        continue
                    else:
                        similarity_output[tk_out.item()] = activations["last_mlp_layer"][id].squeeze(0).cpu()
                    
                all_mlp_activations_and_texts[f'{dataloader_name}_{i}']={
                    "activation": torch.stack(activations['last_mlp_layer'])[:, 0, :].mean(axis=0),
                    "text_original": batch["input_ids"].squeeze(0).cpu(), # Get the single string
                    "output": tokenizer.decode(output_tokens,add_special_tokens=False, skip_special_tokens=True), # Store the model output logits
                    "position": batch["position"][0], 
                    "output_original": generated_output.squeeze(0).cpu(), # Store the generated text
                    "sim_out": similarity_output 
                }    
            except:
                activations.clear()
                act_list.clear()
                continue
            activations.clear()
            act_list.clear()# Clear the activation list for the next batch

    torch.cuda.empty_cache() # Clear CUDA cache to free up memory

    if not all_mlp_activations_and_texts:
        print("No MLP activations collected. Cannot proceed with co-activation graph.")
        exit()

    pickle.dump(all_mlp_activations_and_texts, open(f"{path}/all_mlp_activations_and_texts_{model_name.split('/')[-1]}_{dataloader_name}.pkl", "wb"))
    print("All MLP activations and texts saved to all_mlp_activations_and_texts.pkl")
    return all_mlp_activations_and_texts

def compute_correlation(args):
    i, j, combined_activations = args
    if np.std(combined_activations[:, i]) > 1e-6 and np.std(combined_activations[:, j]) > 1e-6:
        corr, _ = spearmanr(combined_activations[:, i], combined_activations[:, j])
        return (i, j, corr)
    else:
        return (i, j, 0)

def normalize_batch_worker(batch_data):
        results = {}
        for outer_key, inner_dict_sim_out in batch_data:
            if len(inner_dict_sim_out) == 0:
                results[outer_key] = {}
                continue
            
            inner_keys = list(inner_dict_sim_out.keys())
            vectors = [inner_dict_sim_out[k].cpu() for k in inner_keys]
            matrix = torch.stack(vectors)
            normalized_matrix = F.normalize(matrix, p=2, dim=1)
            
            results[outer_key] = {
                inner_key: normalized_matrix[i].cpu()
                for i, inner_key in enumerate(inner_keys)
            }
            
        gc.collect()
        return results

def get_coactivation_graph(all_mlp_activations_and_texts, tokenizer, num_neurons=800, threshold=0.5):
    """
    Combine all MLP activations and texts into a single matrix for co-activation graph generation.
    """
    # Flatten and combine activations
    # Each item in all_mlp_activations_and_texts is a dictionary with "activation" as a tensor
    # We need to reshape each activation tensor to (sequence_length, hidden_size) and stack them
    # This will create a matrix of shape (total_activations, hidden_size)
    
    combined_activations = np.vstack([all_mlp_activations_and_texts[item]["activation"].cpu().reshape(-1, all_mlp_activations_and_texts[item]["activation"].cpu().shape[-1]) for item in all_mlp_activations_and_texts if all_mlp_activations_and_texts[item]['activation'].cpu().dim() >0])
   
    print(f"Collected {combined_activations.shape[0]} activations with {combined_activations.shape[1]} features each.")

    norm = np.linalg.norm(combined_activations, axis=1, keepdims=True)
    normalized_activations = combined_activations / norm
    combined_activations = normalized_activations

    # 4. Co-activation Graph Generator
    num_neurons = combined_activations.shape[1]
    correlation_matrix = np.zeros((num_neurons, num_neurons))

    print("Calculating correlation matrix...")
    subset_neurons = max(100, num_neurons) 

    pairs = [(i, j, combined_activations) for i in range(subset_neurons) for j in range(i, subset_neurons)]

    correlation_matrix = np.zeros((subset_neurons, subset_neurons))

    with multiprocessing.Pool() as pool:
        results = pool.map(compute_correlation, pairs)
        
    for i, j, corr in results:
        correlation_matrix[i, j] = corr
        correlation_matrix[j, i] = corr  # Symmetric matrix

    # Create a graph from the correlation matrix
    G = nx.Graph()
    for i in range(subset_neurons):
        G.add_node(i, label=f"Neuron {i}") # Label can be more meaningful later

    # Only add edges for strong correlations
    for i in range(subset_neurons):
        for j in range(i + 1, subset_neurons):
            if abs(correlation_matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=correlation_matrix[i, j])

    print(f"Co-activation graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (subset of {subset_neurons} neurons).")
    
# 5. Skill Extractor (Enhanced)
    print("\n--- Inferred Concepts/Skills per Neuron ---")

    #normalisation for each input set
    
    # Process each outer_key separately
    result = {}
    print('Normalising activations...')
    
    def normalize_with_batching(data, num_workers=8, batch_size=32):
        all_keys = list(data.keys())
        batches = [
            [(key, data[key]['sim_out']) for key in all_keys[i:i+batch_size]]
            for i in range(0, len(all_keys), batch_size)
        ]
        
        result = {}
        with get_context('spawn').Pool(processes=num_workers) as pool:
            for batch_result in tqdm(
                pool.imap_unordered(normalize_batch_worker, batches),
                total=len(batches)
            ):
                result.update(batch_result)
        
        return result

    result = normalize_with_batching(all_mlp_activations_and_texts, num_workers=40, batch_size=32)
    
    # For each neuron in the subset used for the graph
    #neuron_concepts_input = {}
    print(len(result), len(all_mlp_activations_and_texts))
    neuron_concepts_generated = {}

    # Example input: tokens can repeat across dicts
    dicts_tk_act = []
    print('Aggregating activations per token...')
    # Step 1: Aggregate all activations per token
    for k,v in tqdm(result.items()):
            dicts_tk_act.append(v) # Collect all similarity outputs

    # We'll collect a list of activations for each token
    token_agg = defaultdict(list)
    for d in tqdm(dicts_tk_act):
        for token, vals in d.items():
            token_agg[token].append(vals)

    # Step 2: Flatten and pick the maximum activation per neuron and per token
    # Build a consolidated dict: for each token, pick the max activation value seen per neuron
    print('Getting max activations per token...')
    num_tokens = len(token_agg)
    token_mean_per_neuron = {}
    for token, vectors in token_agg.items():
        arr = np.array(vectors)         # shape: (num_occurrences, num_neurons)
        token_mean_per_neuron[token] = np.median(arr,axis=0) * (arr.shape[0]/num_tokens) #frequency added 

    # Step 3: For each neuron (column), find the token with the highest activation
    tokens = list(token_mean_per_neuron.keys())
    values = np.array([token_mean_per_neuron[k] for k in tokens])  # shape: (num_tokens, num_neurons)

    top1_tokens = []
    for col in range(values.shape[1]):
        max_idx = torch.topk(torch.tensor(values[:, col]),k=10) #top 10
        ids = []
        for i in max_idx.indices:
            ids.append(tokens[i.item()])
            
        top1_tokens.append(ids)
    print('Assigning concepts to neurons...')
    for i, token in enumerate(top1_tokens):
        list_tk = []
        for tk in token:
            list_tk.append(tokenizer.decode(tk, skip_special_tokens=True))
        neuron_concepts_generated[i] = str(list_tk) #Store the top token for each neuron
        G.nodes[i]['token'] = neuron_concepts_generated[i]

    avg_activations = np.mean(combined_activations[:, :subset_neurons], axis=0)
    top_neurons_indices = np.argsort(avg_activations)[::-1][:10]
    print(f"Top 10 most activated neurons (indices within the subset): {top_neurons_indices}")

    G.remove_nodes_from(list(nx.isolates(G)))  # Remove isolated nodes
    return G, neuron_concepts_generated

def print_graph(model_name, G, neuron_concepts_generated, path):
    # 6. Visualization Module
    print("\n--- Visualizing Co-activation Graph ---")

    plt.figure(figsize=(18, 15)) 
    pos = nx.spring_layout(G, k=0.6, iterations=50) # Adjust k and iterations for better spread

    node_labels = {node: neuron_concepts_generated.get(node, f"Neuron {node}") for node in G.nodes()}

    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color="gray")
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=7, font_color="black") # Smaller font size

    plt.title(f"Co-activation Graph of Last MLP Layer Neurons (with Inferred Concepts - Output - {model_name.split('/')[-1]}")
    plt.tight_layout() 
    plt.savefig(f"{path}/co_activation_graph_with_concepts_output_{model_name.split('/')[-1]}.png")
    print("Co-activation graph with concepts saved as co_activation_graph_with_concepts_output.png")
    nx.write_graphml_lxml(G, f"{path}/neuron_output_{model_name.split('/')[-1]}_direct-freq-mul.graphml")
    print("Script finished.")

class MappedUnpickler(pickle.Unpickler):
    def __init__(self, *args, map_location='cpu', **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)
    
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(BytesIO(b), map_location=self._map_location)
        else:
            return super().find_class(module, name)

if __name__ == "__main__":

    # Check if CUDA is available
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

    JOB_DESCRIPTION_DATASET = "lang-uk/recruitment-dataset-job-descriptions-english"
    CV_DATASET = "lang-uk/recruitment-dataset-candidate-profiles-english"

    parser = argparse.ArgumentParser(description="Run inference coactivation script.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", choices=['Qwen/Qwen2.5-0.5B-Instruct','meta-llama/Llama-3.2-1B-Instruct','google/gemma-3-270m-it','mistralai/Mistral-7B-Instruct-v0.2'], help="Model name to use for inference.", required=True)
    parser.add_argument("--num_neurons", type=int, default=896, choices =[896,2048,640,4096], help="Number of neurons to consider for co-activation graph.", required=False)
    parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to generate for each job description and CV.", required=False)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for co-activation graph edges.", required=False)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process from each dataset.", required=False)
    parser.add_argument("--llm_running", type=int, help="1 for running the generation, 0 for the co-activation generation only", default=1)
    parser.add_argument("--activations_path", type=str, default=None, help="Path to pre-saved activations.", required=False)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation. Set 1 for deterministic", required=False)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling for text generation.", required=False)
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling for text generation. Set 1 for deterministic", required=False)
    parser.add_argument("--do_sample", type=int, default=1, help="Whether to use sampling for text generation. Set 0 for deterministic, or 1 for non-deterministic", required=False)
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    num_neurons = args.num_neurons
    max_new_tokens = args.num_tokens
    threshold = args.threshold
    llm_running = args.llm_running
    activations_path = args.activations_path
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    do_sample = False if args.do_sample == 0 else True
    path = args.path

    if llm_running == 1:
        if device is not None:
            tokenizer, model = load_model_and_tokenizer(MODEL_NAME,device)
            job_description_dataloader = load_and_preprocess_data(JOB_DESCRIPTION_DATASET, tokenizer, device, args.num_samples)
            cv_dataloader = load_and_preprocess_data(CV_DATASET, tokenizer, device, args.num_samples)
            
            hook_function(MODEL_NAME, model)

            all_mlp_activations_and_texts_jb = get_activation_map(MODEL_NAME,model, tokenizer, max_new_tokens, job_description_dataloader, 'JB', device, do_sample, top_k, top_p, temperature, path)
            all_mlp_activations_and_texts_cv = get_activation_map(MODEL_NAME,model, tokenizer, max_new_tokens, cv_dataloader, 'CV', device, do_sample, top_k, top_p, temperature, path)

            all_mlp_activations_and_texts = {}
            all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_jb)
            all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_cv)

            G, neuron_concepts_generated = get_coactivation_graph(all_mlp_activations_and_texts, tokenizer, num_neurons, threshold)

            print_graph(MODEL_NAME, G, neuron_concepts_generated, path)

            print(f"Running inference with model: {args.model_name}")
        else:
            tokenizer_jb, model_jb = load_model_and_tokenizer(MODEL_NAME,device_1)
            tokenizer_cv, model_cv = load_model_and_tokenizer(MODEL_NAME,device_2)

            job_description_dataloader = load_and_preprocess_data(JOB_DESCRIPTION_DATASET, tokenizer_jb, device_1, args.num_samples)
            cv_dataloader = load_and_preprocess_data(CV_DATASET, tokenizer_cv, device_2, args.num_samples)
            
            hook_function_jb = hook_function(MODEL_NAME, model_jb)
            hook_function_cv = hook_function(MODEL_NAME, model_cv)

            all_mlp_activations_and_texts_jb = get_activation_map(MODEL_NAME,model_jb, tokenizer_jb, max_new_tokens, job_description_dataloader, 'JB', device_1, do_sample, top_k, top_p, temperature, path)
            all_mlp_activations_and_texts_cv = get_activation_map(MODEL_NAME,model_cv, tokenizer_cv, max_new_tokens, cv_dataloader, 'CV', device_2, do_sample, top_k, top_p, temperature, path)

            all_mlp_activations_and_texts = {}
            all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_jb)
            all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_cv)

            G, neuron_concepts_generated = get_coactivation_graph(all_mlp_activations_and_texts, tokenizer_jb, num_neurons, threshold)

            print_graph(MODEL_NAME, G, neuron_concepts_generated, path)

    elif llm_running == 0 and activations_path is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                with open(f"{path}/all_mlp_activations_and_texts_{MODEL_NAME.split('/')[-1]}_JB.pkl", "rb") as f:
                    all_mlp_activations_and_texts_jb = MappedUnpickler(f, map_location='cuda:0').load()
                with open(f"{path}/all_mlp_activations_and_texts_{MODEL_NAME.split('/')[-1]}_CV.pkl", "rb") as f:
                    all_mlp_activations_and_texts_cv = MappedUnpickler(f, map_location='cuda:1').load()
            else:
                with open(f"{path}/all_mlp_activations_and_texts_{MODEL_NAME.split('/')[-1]}_JB.pkl", "rb") as f:
                    all_mlp_activations_and_texts_jb = MappedUnpickler(f, map_location='cuda:0').load()
                with open(f"{path}/all_mlp_activations_and_texts_{MODEL_NAME.split('/')[-1]}_CV.pkl", "rb") as f:
                    all_mlp_activations_and_texts_cv = MappedUnpickler(f, map_location='cuda:0').load()
        else:
            with open(f"{path}/all_mlp_activations_and_texts_{MODEL_NAME.split('/')[-1]}_JB.pkl", "rb") as f:
                    all_mlp_activations_and_texts_jb = MappedUnpickler(f, map_location='cpu').load()
                
            with open(f"{path}/all_mlp_activations_and_texts_{MODEL_NAME.split('/')[-1]}_CV.pkl", "rb") as f:
                    all_mlp_activations_and_texts_cv = MappedUnpickler(f, map_location='cpu').load()
        
        tokenizer, model = load_model_and_tokenizer(MODEL_NAME,device)

        print('Loading previously saved activations...')
        all_mlp_activations_and_texts = {}
        all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_jb)
        all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_cv)

        G, neuron_concepts_generated = get_coactivation_graph(all_mlp_activations_and_texts, tokenizer, num_neurons, threshold)

        print_graph(MODEL_NAME, G, neuron_concepts_generated, path)

    #reading the new activations generated by the inverse_hidden_features process
    elif llm_running == 0 and activations_path is not None:
        print('Loading previously saved activations...')
        if torch.cuda.is_available():
            with open(f"{path}/{activations_path}_{MODEL_NAME.split('/')[-1]}.pkl", "rb") as f:
                    all_mlp_activations_and_texts = MappedUnpickler(f, map_location='cuda:0').load()
        else:
            with open(f"{path}/{activations_path}_{MODEL_NAME.split('/')[-1]}.pkl", "rb") as f:
                    all_mlp_activations_and_texts = MappedUnpickler(f, map_location='cpu').load()
        
        tokenizer, model = load_model_and_tokenizer(MODEL_NAME,device)
        
        print('Get coactivation graph...')

        G, neuron_concepts_generated = get_coactivation_graph(all_mlp_activations_and_texts, tokenizer, num_neurons, threshold)

        print('Saving graph...')
        print_graph(MODEL_NAME, G, neuron_concepts_generated, path)