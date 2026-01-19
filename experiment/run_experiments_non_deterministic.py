import os
import argparse
import json
from datetime import datetime as dt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", choices=['Qwen/Qwen2.5-0.5B-Instruct','meta-llama/Llama-3.2-1B-Instruct','google/gemma-3-270m-it','mistralai/Mistral-7B-Instruct-v0.2'], help="Model name to use for inference.", required=True)
    parser.add_argument("--num_neurons", type=int, default=896, choices =[896,2048,640,4096], help="Number of neurons to consider for co-activation graph.", required=False)
    parser.add_argument("--num_tokens", type=int, default=100, help="Number of tokens to generate for each job description and CV.", required=False)
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for co-activation graph edges.", required=False)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process from each dataset.", required=False)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation. Set 1 for deterministic", required=False)
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling for text generation.", required=False)
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling for text generation. Set 1 for deterministic", required=False)
    parser.add_argument("--do_sample", type=int, default=1, help="Whether to use sampling for text generation. Set 0 for deterministic, or 1 for non-deterministic", required=False)
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    parser.add_argument("--k", type=int, default=5, help="Topk for the boosting process", required=False)
    parser.add_argument("--boost_factor", type=int, default=2, help="Factor used to boost the neurons", required=False)
    args = parser.parse_args()
    model_name = args.model_name
    num_neurons = args.num_neurons
    num_tokens = args.num_tokens
    threshold = args.threshold
    num_samples = args.num_samples
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    do_sample = args.do_sample
    path = args.path
    k = args.k
    boost_factor = args.boost_factor
    
    start_time = dt.now()

    os.makedirs(path, exist_ok=True)

    with open(f'{path}/experiment_config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    '''
    Assuming that the ONET database is already created, the onet_taxo_kg.py script has been run, generating the onet_esco_knowledge_graph.nt file already created.
    Assuming that the mapping_occupations.py script has been run, generating the position_occupation_map.json file already created.
    '''
    print("Running inference and generating activations...")
    os.system(f"python ../code/inference_coactivation/inference_coactivation.py --model_name {model_name} --num_neurons {num_neurons} --num_samples {num_samples} --threshold {threshold} --path {path}")
    
    print("Running inverse hidden features direct...")
    os.system(f"python ../code/inference_coactivation/inverse_hidden_features_direct.py --model_name {model_name} --path {path}")
    
    print("Generating co-activation graph and token association...")
    os.system(f"python ../code/inference_coactivation/inference_coactivation.py --model_name {model_name} --num_neurons {num_neurons} --num_samples {num_samples} --threshold {threshold} --path {path} --llm_running 0 --activations_path new_activations_direct")
    
    print("Getting top nodes from co-activation graph...")
    os.system(f"python ../code/inference_coactivation/get_top_nodes.py --model_name {model_name} --path {path}")

    print("Mapping the skills and tokens...")
    os.system(f"python ../code/CAG_KG_map/mapping_skills_token.py --model {model_name} --path {path}")

    print("Mapping the skills and output...")
    os.system(f"python ../code/CAG_KG_map/mapping_skills_output.py --model {model_name} --path {path}")

    print("Analyzing the mapping results...")
    os.system(f"python ../code/CAG_KG_map/mapping_analysis.py --model {model_name} --path {path}")

    print("Mapping skills and tokens...")
    os.system(f"python ../code/CAG_KG_map/mapping_skill_all_tokens.py --model {model_name} --path {path}")

    print("Mapping skills and tokens for each instance...")
    os.system(f"python ../code/CAG_KG_map/skills_nodes.py --model {model_name} --path {path}")

    print("Getting nodes to boost...")
    os.system(f"python ../code/inference_coactivation/get_nodes_to_boost.py --model {model_name} --path {path} --k {k}")

    print("Running boost inference and generating activations...")
    os.system(f"python ../code/inference_coactivation/inference_coactivation_weighted.py --model_name {model_name} --num_neurons {num_neurons} --num_samples {num_samples} --threshold {threshold} --path {path} --k {5} --boost_factor {boost_factor}")

    print("Mapping the skills and output...")
    os.system(f"python ../code/CAG_KG_map/mapping_skills_output.py --model {model_name} --path {path} --weighted 1 --k {k}")

    print("Analyzing the mapping results...")
    os.system(f"python ../code/CAG_KG_map/mapping_analysis.py --model {model_name} --path {path} --weighted 1 --k {k}")

    end_time = dt.now()

    elapsed = end_time - start_time
    process_time_str ="Process took: %02d:%02d:%02d:%02d" % (elapsed.days, elapsed.seconds // 3600, elapsed.seconds // 60 % 60, elapsed.seconds % 60)

    with open(f'{path}/experiment_config.json', 'r') as feedsjson:
        feeds = json.load(feedsjson)

    feeds['timestamp']=process_time_str
    with open(f'{path}/experiment_config.json', 'w') as f:
        f.write(json.dumps(feeds, indent=4))
    print(process_time_str)
