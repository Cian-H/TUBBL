import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from datasets import load_dataset
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
import pandas as pd
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
    
class MLPNeuronBooster:
    """
    Context manager to temporarily boost a specific neuron in the MLP layer
    during generation without modifying the model permanently.
    """
    def __init__(self, model, neuron_idx, boost_factor=2.0):
        """
        Args:
            model: The HuggingFace model
            layer_idx: Index of the transformer layer (0-based)
            neuron_idx: Index of the neuron in the MLP intermediate layer
            boost_factor: Multiplier for the neuron's activation (e.g., 2.0 = 2x)
        """
        self.model = model
        self.neuron_idx = neuron_idx
        self.boost_factor = boost_factor
        self.hook_handles = []

    def _create_hook(self):
        """Create a forward hook to boost the neuron activation"""
        def hook(module, input, output):
            # output shape: (batch_size, seq_len, intermediate_size)
            if isinstance(output, tuple):
                output = output[0]
            
            for idx in self.neuron_idx:
            # Boost the specific neuron across all batch and sequence positions
                output[..., int(idx)] *= self.boost_factor
                output[..., int(idx)] = torch.abs(output[..., int(idx)])
            return output
        
        return hook

    def __enter__(self):
        """Register the hook on entry"""
        # Access the MLP intermediate layer (usually mlp.dense_h_to_4h or similar)

        last_mlp_layer = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer in reversed(self.model.model.layers):       
                last_mlp_layer = layer
                break

        hook = self._create_hook()
        handle = last_mlp_layer.register_forward_hook(hook)
        self.hook_handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the hook on exit"""
        for handle in self.hook_handles:
            handle.remove()

def get_activation_map(model_name, model, tokenizer, max_new_tokens, dataloader, dataloader_name, device, do_sample, top_k, top_p, temperature, path, df, boost_factor=2.0):
    all_mlp_activations_and_texts = {}

    print(f"Processing {dataloader_name}...")
    for i, batch in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            # Generate text from the model
            df_row = df[df['id'] == f'{dataloader_name}_{i}']
            if len(df_row) == 0:
                continue
            else:
                neuron_idx = df_row['top10_nodes_skills'].values[0]
                boost_factor = boost_factor
                with MLPNeuronBooster(model, neuron_idx, boost_factor):
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
                    
                    all_mlp_activations_and_texts[f'{dataloader_name}_{i}']={
                        "activation_w": torch.stack(activations['last_mlp_layer'])[:, 0, :].mean(axis=0),
                        "output_w": tokenizer.decode(output_tokens,add_special_tokens=False, skip_special_tokens=True), # Store the model output logits
                        "position_w": batch["position"][0], 
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

    pickle.dump(all_mlp_activations_and_texts, open(f"{path}/all_mlp_activations_and_texts_weight_{model_name.split('/')[-1]}_{dataloader_name}.pkl", "wb"))
    print("All MLP activations and texts saved to all_mlp_activations_and_texts.pkl")
    return all_mlp_activations_and_texts


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
    # %%

    parser = argparse.ArgumentParser(description="Run inference coactivation script.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", choices=['Qwen/Qwen2.5-0.5B-Instruct','meta-llama/Llama-3.2-1B-Instruct','google/gemma-3-270m-it','mistralai/Mistral-7B-Instruct-v0.2'], help="Model name to use for inference.", required=True)
    parser.add_argument("--num_neurons", type=int, default=896, choices =[896,2048,640,4096], help="Number of neurons to consider for co-activation graph.", required=False)
    parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to generate for each job description and CV.", required=False)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process from each dataset.", required=False)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation. Set 1 for deterministic", required=False)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling for text generation.", required=False)
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling for text generation. Set 1 for deterministic", required=False)
    parser.add_argument("--do_sample", type=int, default=1, help="Whether to use sampling for text generation. Set 0 for deterministic, or 1 for non-deterministic", required=False)
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    parser.add_argument("--k", type=int, default=10, help="Number of top nodes to select.", required=False)
    parser.add_argument("--boost_factor", type=float, default=2.0, help="Factor to boost neuron activations during generation.", required=False)
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    num_neurons = args.num_neurons 
    max_new_tokens = args.num_tokens
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    do_sample = False if args.do_sample == 0 else True
    path = args.path
    k = args.k
    boost_factor = args.boost_factor

    df = pickle.load(open(f'{path}df_direct_all_top{k}_nodes_skills_{MODEL_NAME.split("/")[-1]}.pkl', 'rb'))
  
    if device is not None:
        tokenizer, model = load_model_and_tokenizer(MODEL_NAME,device)
        job_description_dataloader = load_and_preprocess_data(JOB_DESCRIPTION_DATASET, tokenizer, device, args.num_samples)
        cv_dataloader = load_and_preprocess_data(CV_DATASET, tokenizer, device, args.num_samples)
        
        hook_function(MODEL_NAME, model)

        all_mlp_activations_and_texts_jb = get_activation_map(MODEL_NAME,model, tokenizer, max_new_tokens, job_description_dataloader, 'JB', device, do_sample, top_k, top_p, temperature, path, df, boost_factor)
        all_mlp_activations_and_texts_cv = get_activation_map(MODEL_NAME,model, tokenizer, max_new_tokens, cv_dataloader, 'CV', device, do_sample, top_k, top_p, temperature, path, df, boost_factor)

        all_mlp_activations_and_texts = {}
        all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_jb)
        all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_cv)

        df_activations = pd.DataFrame.from_dict(all_mlp_activations_and_texts, columns=['activation_w','output_w','position_w'], orient='index').reset_index().rename(columns={'index':'id'})
        df_activations['activation_w'] = df_activations['activation_w'].apply(lambda x: x.cpu())
        df_joined = df.merge(df_activations, on='id', how='inner')

        df_joined.to_pickle(f'{path}df_direct_all_top{k}_nodes_skills_with_outputs_{MODEL_NAME.split("/")[-1]}_weighted.pkl')

        print(f"Running inference with model: {args.model_name}")

    else:
        tokenizer_jb, model_jb = load_model_and_tokenizer(MODEL_NAME,device_1)
        tokenizer_cv, model_cv = load_model_and_tokenizer(MODEL_NAME,device_2)

        job_description_dataloader = load_and_preprocess_data(JOB_DESCRIPTION_DATASET, tokenizer_jb, device_1, args.num_samples)
        cv_dataloader = load_and_preprocess_data(CV_DATASET, tokenizer_cv, device_2, args.num_samples)
        
        hook_function_jb = hook_function(MODEL_NAME, model_jb)
        hook_function_cv = hook_function(MODEL_NAME, model_cv)

        all_mlp_activations_and_texts_jb = get_activation_map(MODEL_NAME,model_jb, tokenizer_jb, max_new_tokens, job_description_dataloader, 'JB', device_1, do_sample, top_k, top_p, temperature, path, df,boost_factor)
        all_mlp_activations_and_texts_cv = get_activation_map(MODEL_NAME,model_cv, tokenizer_cv, max_new_tokens, cv_dataloader, 'CV', device_2, do_sample, top_k, top_p, temperature, path, df, boost_factor)


        all_mlp_activations_and_texts = {}
        all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_jb)
        all_mlp_activations_and_texts.update(all_mlp_activations_and_texts_cv)

        df_activations = pd.DataFrame.from_dict(all_mlp_activations_and_texts, columns=['activation_w','output_w','position_w'], orient='index').reset_index().rename(columns={'index':'id'})
        df_activations['activation_w'] = df_activations['activation_w'].apply(lambda x: x.cpu())
        df_joined = df.merge(df_activations, on='id', how='inner')

        df_joined.to_pickle(f'{path}df_direct_all_top{k}_nodes_skills_with_outputs_{MODEL_NAME.split("/")[-1]}_weighted.pkl')