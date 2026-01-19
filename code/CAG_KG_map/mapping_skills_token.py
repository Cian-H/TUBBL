import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import ast
from functools import lru_cache
from joblib import Parallel, delayed
from tqdm import tqdm
import difflib
from functools import partial
import argparse

class StringSimilarityMatcher:
    """
    Ultra-fast string-based matching without embeddings.
    Use this if you have controlled vocabularies (skills & tokens are similar words/tokens).
    """
    
    def compute_string_similarity(skills, tokens, threshold=0.6):
        """
        Use string similarity (e.g., SequenceMatcher) instead of embeddings.
        ~100-1000x faster than semantic embeddings for controlled vocabularies.
        """
        if not skills or not tokens:
            return {
                'threshold_met': False,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'above_threshold_count': 0,
                'total_comparisons': 0,
                'skills_with_matches': []
            }
        
        similarity_scores = []
        skill_has_match = []
        
        for skill in skills:
            skill_lower = skill.lower()
            max_sim = 0.0
            
            for token in tokens:
                token_lower = str(token).lower()
                # Option 1: Token-in-skill substring matching (fastest)
                if token_lower in skill_lower or skill_lower in token_lower:
                    sim = 1.0
                else:
                    # Option 2: SequenceMatcher ratio (more flexible)
                    sim = difflib.SequenceMatcher(None, skill_lower, token_lower).ratio()
                
                similarity_scores.append(sim)
                max_sim = max(max_sim, sim)
            
            skill_has_match.append(max_sim >= threshold)
        
        similarity_array = np.array(similarity_scores)
        above_threshold = (similarity_array >= threshold).sum()
        total_comparisons = len(similarity_scores)
        
        return {
            'threshold_met': bool(np.all(skill_has_match)),
            'avg_similarity': float(np.mean(similarity_array)) if similarity_array.size > 0 else 0.0,
            'max_similarity': float(np.max(similarity_array)) if similarity_array.size > 0 else 0.0,
            'above_threshold_count': int(above_threshold),
            'total_comparisons': int(total_comparisons),
            'skills_with_matches': sum(1 for i in skill_has_match if i == True)
        }


def process_single_job_string_match(row_data, job_skills_data, threshold=0.5):
    """Process using ultra-fast string matching."""
    idx, job_title, tokens = row_data
    
    if isinstance(tokens, str):
        tokens = [ast.literal_eval(token) if token.startswith('[') and token.endswith(']') 
                 else token for token in tokens]
    
    skills = job_skills_data.get('skills',[])
    
    results = []
    for token_list in tokens:
        token_list = ast.literal_eval(token_list)

        similarity_results = StringSimilarityMatcher.compute_string_similarity(
            skills, token_list, threshold=threshold
        )
        
        results.append({
            'job_title': job_title,
            'num_skills': len(skills),
            'num_tokens': len(token_list),
            'threshold_met': similarity_results['threshold_met'],
            'avg_similarity': similarity_results['avg_similarity'],
            'max_similarity': similarity_results['max_similarity'],
            'above_threshold_count': similarity_results['above_threshold_count'],
            'skills_with_matches': similarity_results['skills_with_matches'],
            'total_comparisons': similarity_results['total_comparisons'],
            'above_threshold_percentage': (similarity_results['above_threshold_count'] / 
                                          similarity_results['total_comparisons'] * 100 
                                          if similarity_results['total_comparisons'] > 0 else 0)
        })
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mapping skills and tokens.")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name')
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    args = parser.parse_args()
    model_name = args.model_name
    path = args.path
    
    # Load data
    df = pickle.load(open(f'{path}/df_direct_freq_mul_{model_name.split("/")[-1]}.pkl', 'rb'))
    with open(f'../code/CAG_KG_map/position_occupation_map.json', 'r') as f:
        job_skills_data = json.load(f)
    
    rows_data = [(idx, row['id'], row['top100_tokens']) for idx, row in df.iterrows()]

    results_list = Parallel(n_jobs=-1)(
        delayed(process_single_job_string_match)(row,job_skills_data[row[1]],0.3) 
        for row in tqdm(rows_data, desc="Processing (String Match)")
    )
    
    all_results_b = [item for sublist in results_list for item in sublist]
    results_df_b = pd.DataFrame(all_results_b)
    results_df_b.to_pickle(f'{path}/skill_token_similarity_results_string_{model_name.split("/")[-1]}.pkl')
    print(f"Completed: {len(results_df_b)} results")