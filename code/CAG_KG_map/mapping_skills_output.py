import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from typing import List, Dict, Set, Tuple, Optional
from joblib import Parallel, delayed
import pickle
import json
from tqdm import tqdm
import argparse

class SkillMatcherSentenceEncoder:
    """
    Matches skills using pre-trained sentence encoders.
    
    Uses semantic embeddings to find relevant skills from LLM output,
    capturing meaning beyond simple keyword matching.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize with a pre-trained sentence encoder.
        
        Args:
            model_name: HuggingFace model identifier. Options:
                - 'all-MiniLM-L6-v2': Fast, good quality, 384 dims 
                - 'all-mpnet-base-v2': Slower, higher quality, 768 dims
                - 'paraphrase-MiniLM-L6-v2': Specialized for paraphrasing
                - 'multi-qa-MiniLM-L6-cos-v1': Good for retrieval/search
                - 'all-distilroberta-v1': Fast, good for general similarity
        
        The first load downloads the model (~40-400MB depending on model).
        Subsequent loads are cached locally.
        """
        self.model_name = model_name
        print(f"Loading sentence encoder: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        
        self.output_embedding = None
        self.skill_embeddings = None
        self.kg_skills = None
        self.similarity_scores = None
    
    def fit_and_score(self, llm_output: str, kg_skills: List[str]) -> Dict:
        """
        Generate embeddings and calculate similarity scores.
        
        Args:
            llm_output: Full LLM output text containing inferred skills
            kg_skills: List of skills from knowledge graph
        
        Returns:
            Dictionary with similarity scores and ranked skills
        """
        self.kg_skills = kg_skills
        
        # Encode the LLM output
        #print(f"Encoding LLM output ({len(llm_output)} characters)...")
        self.output_embedding = self.model.encode(
            llm_output,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Encode all skills
        #print(f"Encoding {len(kg_skills)} skills...")
        self.skill_embeddings = self.model.encode(
            kg_skills,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        # Calculate cosine similarity between LLM output and each skill
        #print("Calculating cosine similarities...")
        similarities = util.pytorch_cos_sim(
            self.output_embedding,
            self.skill_embeddings
        )
        
        # Convert to numpy
        self.similarity_scores = similarities[0].cpu().numpy()
        
        # Create results dictionary
        results = []
        for idx, skill in enumerate(kg_skills):
            results.append({
                'skill': skill,
                'similarity_score': float(self.similarity_scores[idx]),
                'idx': idx
            })
        
        # Sort by similarity score (highest first)
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        match_skills = self.match_skills(threshold=0.3)
        return {
            'similarity_results': results,
            'num_skills': len(kg_skills),
            'matched_skills_count': len(match_skills),
            '%_matched_skills': (len(match_skills) / len(kg_skills) * 100) if len(kg_skills) > 0 else 0.0 #coverage
        }
    
    def match_skills(self, threshold: float) -> List[str]:
        """
        Extract skills above similarity threshold.
        
        Args:
            threshold: Similarity threshold (0.0 to 1.0)
        
        Returns:
            List of matched skills
        """
        if self.similarity_scores is None:
            raise ValueError("Must call fit_and_score() first")
        
        matched = []
        for idx, skill in enumerate(self.kg_skills):
            if self.similarity_scores[idx] >= threshold:
                matched.append(skill)
        
        return matched
    
    def get_similarity_for_skill(self, skill: str) -> Optional[float]:
        """
        Get similarity score for a specific skill.
        
        Args:
            skill: The skill to query
        
        Returns:
            Similarity score or None if skill not found
        """
        if self.kg_skills is None:
            raise ValueError("Must call fit_and_score() first")
        
        if skill not in self.kg_skills:
            return None
        
        idx = self.kg_skills.index(skill)
        return float(self.similarity_scores[idx])
    
    def get_similar_skills(self, skill: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find skills most similar to a given skill.
        
        Useful for understanding skill relationships in your knowledge graph.
        
        Args:
            skill: The reference skill
            top_k: Number of similar skills to return
        
        Returns:
            List of (skill_name, similarity_score) tuples
        """
        if self.kg_skills is None:
            raise ValueError("Must call fit_and_score() first")
        
        if skill not in self.kg_skills:
            return []
        
        skill_idx = self.kg_skills.index(skill)
        skill_embedding = self.skill_embeddings[skill_idx:skill_idx+1]
        
        # Calculate similarity with all other skills
        similarities = util.pytorch_cos_sim(skill_embedding, self.skill_embeddings)[0]
        
        # Get top-k most similar
        top_indices = torch.topk(similarities, k=min(top_k+1, len(self.kg_skills)))[1]
        
        results = []
        for idx in top_indices:
            idx_val = idx.item()
            if self.kg_skills[idx_val] != skill:  # Exclude the skill itself
                results.append((
                    self.kg_skills[idx_val],
                    float(similarities[idx_val])
                ))
        
        return results[:top_k]
    
    def get_embedding_similarity_matrix(self) -> np.ndarray:
        """
        Get the full similarity matrix between all skills.
        
        Returns:
            NxN matrix where N = number of skills
        """
        if self.skill_embeddings is None:
            raise ValueError("Must call fit_and_score() first")
        
        similarities = util.pytorch_cos_sim(self.skill_embeddings, self.skill_embeddings)
        return similarities.cpu().numpy()


def process_single_row(row, job_skills_data, matcher):
    """
    Process a single row for fit_and_score.
    
    Args:
        row: Single row from rows_data
        job_skills_data: Dictionary containing job skills
        matcher: SkillMatcherSentenceEncoder instance
    
    Returns:
        Tuple of (job_id, result) or None if skipped
    """
    llm_output = row[-1]
    job_id = row[1]
    kg_skills = job_skills_data[job_id]['skills']
    
    # Skip if no skills or empty output
    if len(kg_skills) == 0 or len(llm_output.strip()) == 0:
        return None
    
    # Process this row
    result = matcher.fit_and_score(llm_output, kg_skills)
    return (job_id, result)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mapping skills and output.")
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name')
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    parser.add_argument("--weighted", type=int, default=0, help="Whether to use weighted activations.", required=False)
    parser.add_argument("--k", type=int, default=10, help="Number of top nodes to select.", required=False)
    args = parser.parse_args()
    model_name = args.model_name
    weighted = args.weighted
    path = args.path
    k = args.k
    
    # Load data
    with open(f'../code/CAG_KG_map/position_occupation_map.json', 'r') as f:
        job_skills_data = json.load(f)
    
    if weighted == 0:
        df = pickle.load(open(f'{path}/df_direct_freq_mul_{model_name.split("/")[-1]}.pkl', 'rb'))
        rows_data = [(idx, row['id'], row['output']) for idx, row in df.iterrows()]
    else:
        df = pickle.load(open(f'{path}/df_direct_all_top{k}_nodes_skills_with_outputs_{model_name.split("/")[-1]}_weighted.pkl', 'rb'))
        rows_data = [(idx, row['id'], row['output_w']) for idx, row in df.iterrows()]

    # Use lightweight, fast model
    matcher = SkillMatcherSentenceEncoder()

    
    result={}
    for row in tqdm(rows_data, desc="Processing Fit and Score"):
        llm_output = row[-1]
        kg_skills = job_skills_data[row[1]]['skills']
        if len(kg_skills) > 0 and len(llm_output.strip())>0:
            result[row[1]]=matcher.fit_and_score(llm_output, kg_skills)
        else:
            continue
    
    if weighted == 0:
        pickle.dump(result,open(f'{path}/sentence_encoder_skill_matching_results_{model_name.split("/")[-1]}.pkl','wb'))
    else:
        pickle.dump(result,open(f'{path}/sentence_encoder_skill_matching_results_{model_name.split("/")[-1]}_top{k}_weighted.pkl','wb'))