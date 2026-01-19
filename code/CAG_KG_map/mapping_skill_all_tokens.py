import json
import ast
from typing import Dict, List, Set, Tuple
import networkx as nx
from multiprocessing import Pool, cpu_count
from difflib import SequenceMatcher
import pickle
import sys
import argparse
from pathlib import Path

# ============================================================================
# PARALLELIZED SKILL-NODE MATCHING SCRIPT
# ============================================================================

def parse_token_string(token_str: str) -> List[str]:
    """
    Convert string representation of list to actual Python list.
    Handles both formats: '["a", "b"]' and "['a', 'b']"
    """
    try:
        return ast.literal_eval(token_str)
    except (ValueError, SyntaxError):
        # Fallback: try to split by common delimiters
        return [t.strip() for t in token_str.strip('[]').split(',')]


def token_similarity(token_list: List[str], skill: str, threshold: float = 0.6) -> float:
    """
    Calculate similarity between a skill string and a list of tokens.
    Returns max similarity ratio found between skill and any token.
    
    Matching strategy:
    1. Exact match (case-insensitive) → 1.0
    2. Substring match → 0.85
    3. Fuzzy match via SequenceMatcher → ratio (if >= threshold)
    """
    max_ratio = 0.0
    skill_lower = skill.lower()
    
    for token in token_list:
        token_lower = token.lower()
        # Check exact match first (fastest)
        if token_lower == skill_lower:
            return 1.0
        # Check substring match
        if skill_lower in token_lower or token_lower in skill_lower:
            return 0.85
        # Use SequenceMatcher for fuzzy matching
        ratio = SequenceMatcher(None, skill_lower, token_lower).ratio()
        max_ratio = max(max_ratio, ratio)
    
    return max_ratio if max_ratio >= threshold else 0.0


def match_skill_to_nodes(args: Tuple[str, List, Dict]) -> Tuple[str, List]:
    """
    Worker function: Match a single skill to nodes in the graph.
    
    Args:
        args: (skill, node_ids, node_token_dict)
    
    Returns:
        (skill, list of matching node_ids)
    """
    skill, node_ids, node_token_dict = args
    matching_nodes = []
    similarity_score=[]

    for node_id in node_ids:
        tokens = node_token_dict[node_id]
        if token_similarity(tokens, skill) >= 0.07:
            matching_nodes.append(node_id)
            similarity_score.append(token_similarity(tokens, skill))
    
    similarity_sorted_idx = sorted(range(len(similarity_score)), key=lambda k: similarity_score[k])
    matching_nodes_sorted = [matching_nodes[i] for i in similarity_sorted_idx[:4]]
    return (skill, matching_nodes_sorted)
    #return (skill, matching_nodes)


def process_json_entry(args: Tuple[str, Dict, Dict[str, List]]) -> Tuple[str, Dict]:
    """
    Worker function: Process a single JSON entry and map skills to nodes.
    
    Args:
        args: (key, json_entry, skill_to_nodes_dict)
    
    Returns:
        (key, updated_entry_with_skill_node_mapping)
    """
    key, entry, skill_to_nodes = args
    
    # Create mapping of skills to their corresponding nodes
    skill_node_mapping = {}
    for skill in entry.get('skills', []):
        if skill in skill_to_nodes:
            skill_node_mapping[skill] = skill_to_nodes[skill]
        else:
            skill_node_mapping[skill] = []
    
    # Create new entry with skill_node mapping
    new_entry = entry.copy()
    new_entry['skill_node'] = skill_node_mapping
    
    return (key, new_entry)


def parallelize_skill_node_matching(
    json_file_path: str,
    graph_pickle_path: str,
    output_json_path: str,
    n_processes: int = None,
    similarity_threshold: float = 0.6,
    verbose: bool = True
) -> Dict:
    """
    Main function: Parallelize skill-to-node matching process.
    
    Args:
        json_file_path: Path to JSON file with skills
        graph_pickle_path: Path to pickled NetworkX graph
        output_json_path: Path where to save output JSON
        n_processes: Number of processes (default: CPU count)
        similarity_threshold: Similarity threshold for matching (0.0 to 1.0)
        verbose: Print progress messages
    
    Returns:
        Updated JSON dictionary with skill_node mappings
    
    Example:
        result = parallelize_skill_node_matching(
            json_file_path='skills.json',
            graph_pickle_path='graph.pkl',
            output_json_path='output.json',
            n_processes=4,
            similarity_threshold=0.6
        )
    """
    
    n_processes = n_processes or cpu_count()
    if verbose:
        print(f"Using {n_processes} processes")
    
    # ========== STEP 1: Load data ==========
    if verbose:
        print("\n[Step 1] Loading JSON and Graph...")
 
    json_data = json_file_path
    graph = graph_pickle_path
    
    if verbose:
        print(f"Loaded {len(json_data)} entries from JSON")
        print(f"Loaded graph with {graph.number_of_nodes()} nodes")
    
    # ========== STEP 2: Extract and parse node tokens ==========
    if verbose:
        print("\n[Step 2] Parsing node tokens...")
    node_token_dict = {}
    node_ids = []
    
    for node_id, attr in graph.nodes(data=True):
        token_str = attr.get('token', '[]')
        tokens = parse_token_string(token_str)
        node_token_dict[node_id] = tokens
        node_ids.append(node_id)
    
    if verbose:
        print(f"Parsed tokens for {len(node_token_dict)} nodes")
    
    # ========== STEP 3: Collect all unique skills ==========
    if verbose:
        print("\n[Step 3] Collecting unique skills...")
    all_skills = set()
    for entry in json_data.values():
        all_skills.update(entry.get('skills', []))
    
    if verbose:
        print(f"Found {len(all_skills)} unique skills")
    
    # ========== STEP 4: Parallelize skill-to-node matching ==========
    if verbose:
        print(f"\n[Step 4] Matching skills to nodes (parallel with {n_processes} processes)...")
    
    # Prepare arguments for parallel processing
    skill_args = [
        (skill, node_ids, node_token_dict)
        for skill in all_skills
    ]
    
    skill_to_nodes = {}
    with Pool(n_processes) as pool:
        results = pool.imap_unordered(match_skill_to_nodes, skill_args, chunksize=10)
        for skill, matching_nodes in results:
            skill_to_nodes[skill] = matching_nodes
            if verbose:
                if matching_nodes:
                    print(f"  ✓ '{skill}' → nodes {matching_nodes}")
                else:
                    print(f"  ✗ '{skill}' → no matches")
    
    # ========== STEP 5: Parallelize JSON entry processing ==========
    if verbose:
        print(f"\n[Step 5] Processing JSON entries (parallel with {n_processes} processes)...")
    
    # Prepare arguments for parallel processing
    json_args = [
        (key, entry, skill_to_nodes)
        for key, entry in json_data.items()
    ]
    
    updated_json = {}
    with Pool(n_processes) as pool:
        results = pool.imap_unordered(process_json_entry, json_args, chunksize=5)
        for key, updated_entry in results:
            updated_json[key] = updated_entry
            if verbose:
                print(f"  ✓ Processed '{key}'")
    
    # ========== STEP 6: Save output ==========
    if verbose:
        print(f"\n[Step 6] Saving output JSON to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(updated_json, f, indent=2)
    
    if verbose:
        print("✓ Complete!")
    
    return updated_json


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mapping skill-node.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", choices=['Qwen/Qwen2.5-0.5B-Instruct','meta-llama/Llama-3.2-1B-Instruct','google/gemma-3-270m-it','mistralai/Mistral-7B-Instruct-v0.2'], help="Model name to use for inference.", required=True)
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)

    args = parser.parse_args()

    model_name = args.model_name
    path = args.path

    graph_type = 'direct-freq-mul'
    new_path = Path(path).cwd().parents[0]
    new_path = str(new_path)+'/code/CAG_KG_map/'
    
    co_activation = nx.read_graphml(f'{path}/neuron_output_{model_name.split("/")[-1]}_{graph_type}.graphml')
    with open(f'{new_path}position_occupation_map.json', 'r') as f:
        job_skills_data = json.load(f)

    result = parallelize_skill_node_matching(
        json_file_path=job_skills_data,
        graph_pickle_path=co_activation,
        output_json_path=path+f'/skill_all_nodes_{model_name.split("/")[-1]}.json',
        similarity_threshold=0.7,
        verbose=False
    )