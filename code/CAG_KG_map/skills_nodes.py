import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Set, List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import time
import psutil
import argparse
from pathlib import Path

try:
    import orjson  # Faster JSON parsing
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
    import json as orjson  # Fallback to standard json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_system_resources():
    """Check available RAM and CPU cores."""
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    cpu_cores = cpu_count()
    
    logger.info(f" System Resources:")
    logger.info(f"   Available RAM: {available_ram_gb:.2f} GB")
    logger.info(f"   CPU Cores: {cpu_cores}")
    
    return available_ram_gb, cpu_cores


def estimate_memory_usage(nodes_json_path: str) -> float:
    """Estimate memory needed to load nodes JSON in RAM."""
    file_size_gb = Path(nodes_json_path).stat().st_size / (1024**3)
    estimated_ram_gb = file_size_gb * 1.8
    
    logger.info(f" Nodes JSON Analysis:")
    logger.info(f"   File size: {file_size_gb:.2f} GB")
    logger.info(f"   Estimated RAM needed (1.8x): {estimated_ram_gb:.2f} GB")
    
    return estimated_ram_gb, file_size_gb


def recommend_strategy(file_size_gb: float, available_ram_gb: float) -> str:
    """Recommend loading strategy based on system resources."""
    
    if available_ram_gb > file_size_gb * 2.5:
        strategy = 'full'
        logger.info(f" Recommended: 'full' (fastest)")
    elif available_ram_gb > file_size_gb * 1.2:
        strategy = 'batch'
        logger.info(f" Recommended: 'batch' (balanced)")
    else:
        strategy = 'streaming'
        logger.warning(f"  Recommended: 'streaming' (memory-constrained)")
    
    return strategy


# ============================================================================
# NODE LOADING STRATEGIES
# ============================================================================

def load_nodes_json_full(nodes_json_path: str) -> Dict:
    """
    Load entire nodes JSON file into memory for fast O(1) lookups.
    
    Args:
        nodes_json_path: Path to the nodes JSON file
        
    Returns:
        Dictionary with ID -> skill_nodes mapping
        
    Note: This trades memory for speed. If file is huge (>8GB),
          use load_nodes_json_batch() instead.
    """
    logger.info(f" Loading entire nodes JSON into memory...")
    logger.info(f"   Path: {nodes_json_path}")
    
    start_time = time.time()
    
    try:
        if HAS_ORJSON:
            with open(nodes_json_path, 'rb') as f:
                nodes_dict = orjson.loads(f.read())
            logger.info(" Loaded using orjson (fast)")
        else:
            with open(nodes_json_path, 'r') as f:
                nodes_dict = json.load(f)
            logger.info(" Loaded using standard json")
            
        elapsed = time.time() - start_time
        logger.info(f" Loaded {len(nodes_dict):,} ID entries in {elapsed:.2f}s")
        return nodes_dict
        
    except MemoryError as e:
        logger.error(" NOT ENOUGH MEMORY to load full nodes JSON!")
        logger.info(" Switch to batch loading strategy")
        raise


def load_nodes_json_batch(nodes_json_path: str, required_ids: Set[str]) -> Dict:
    """
    Stream nodes JSON and load ONLY required IDs (memory-efficient).
    
    Args:
        nodes_json_path: Path to the nodes JSON file
        required_ids: Set of ID strings to extract
        
    Returns:
        Dictionary with only required IDs and their nodes
        
    Use this if full nodes JSON won't fit in memory.
    """
    logger.info(f" Batch loading nodes JSON...")
    logger.info(f"   Required IDs: {len(required_ids):,}")
    logger.info(f"   Path: {nodes_json_path}")
    
    start_time = time.time()
    nodes_dict = {}
    loaded_count = 0
    
    try:
        if HAS_ORJSON:
            with open(nodes_json_path, 'rb') as f:
                full_dict = orjson.loads(f.read())
        else:
            with open(nodes_json_path, 'r') as f:
                full_dict = json.load(f)
        
        # Extract only required IDs
        for id_str in required_ids:
            if id_str in full_dict:
                nodes_dict[id_str] = full_dict[id_str]
                loaded_count += 1
        
        elapsed = time.time() - start_time
        logger.info(f" Loaded {loaded_count:,}/{len(required_ids):,} required IDs in {elapsed:.2f}s")
        logger.info(f"   Coverage: {100 * loaded_count / len(required_ids):.1f}%")
        
        return nodes_dict
        
    except MemoryError as e:
        logger.error(" NOT ENOUGH MEMORY for batch loading!")
        logger.info(" Use streaming strategy (slower but minimal memory)")
        raise


# ============================================================================
# ROW PROCESSING FUNCTIONS
# ============================================================================

def process_row_optimized(args) -> Tuple[int, List, List]:
    """
    Process a single row using pre-loaded nodes dictionary (O(1) lookup).
    
    Args:
        args: Tuple of (index, id_value, skills_dict, nodes_dict)
    
    Returns:
        Tuple of (index, skills_list, nodes_list)
    """
    idx, id_value, skills_dict, nodes_dict = args
    id_str = str(id_value)
    
    # Get skills for this ID (safely)
    if id_str not in skills_dict:
        return idx, [], []
    
    try:
        skills = skills_dict[id_str].get('skills', [])
    except (KeyError, TypeError):
        return idx, [], []
    
    skills_set = set(skills)
    nodes_set = set()
    
    # Get nodes for this ID (O(1) lookup instead of streaming entire file!)
    if id_str in nodes_dict:
        try:
            skill_node = nodes_dict[id_str].get('skill_node', {})
            for skill in skills:
                if skill in skill_node:
                    nodes_set.update(skill_node[skill])
        except (KeyError, TypeError):
            pass
    
    return idx, skills, list(nodes_set)


# ============================================================================
# MAIN PROCESSOR CLASS
# ============================================================================

class SkillNodeProcessor:
    """
    Optimized processor that loads nodes data once instead of streaming per-row.
    Supports multiple loading strategies for different memory constraints.
    
    Strategies:
    - 'full': Load entire nodes JSON (fastest, needs RAM)
    - 'batch': Load only required IDs (balanced)
    - 'streaming': Stream per-row (slowest, minimum RAM)
    """

    def __init__(
        self,
        skills_dict: Dict,
        nodes_json_path: str,
        loading_strategy: str = 'batch'
    ):
        """
        Initialize the processor.

        Args:
            skills_dict: Dictionary with ID -> skills mapping
            nodes_json_path: Path to nodes JSON file
            loading_strategy: 'full', 'batch', or 'streaming'
        """
        self.loading_strategy = loading_strategy
        self.skills_dict = skills_dict
        self.nodes_json_path = nodes_json_path
        self.nodes_dict = None
        self.load_time = None
        
        logger.info(f" Initialized SkillNodeProcessor")
        logger.info(f"   Skill IDs: {len(self.skills_dict):,}")
        logger.info(f"   Strategy: '{loading_strategy}'")

    def load_nodes(self, required_ids: Set[str] = None) -> float:
        """
        Pre-load nodes data using selected strategy.
        
        Args:
            required_ids: Set of IDs to load (for batch strategy)
            
        Returns:
            Time taken to load (seconds)
        """
        start_time = time.time()
        
        if self.loading_strategy == 'full':
            self.nodes_dict = load_nodes_json_full(self.nodes_json_path)
            
        elif self.loading_strategy == 'batch':
            if required_ids is None:
                required_ids = set(self.skills_dict.keys())
            self.nodes_dict = load_nodes_json_batch(
                self.nodes_json_path, 
                required_ids
            )
        else:
            logger.warning("  Using streaming strategy (slower)")
            self.nodes_dict = {}
        
        self.load_time = time.time() - start_time
        return self.load_time

    def process_dataframe_parallel(
        self,
        df: pd.DataFrame,
        id_column: str = 'id',
        n_jobs: int = None,
        preload_nodes: bool = True
    ) -> pd.DataFrame:
        """
        Process dataframe in parallel with optimized node loading.

        Args:
            df: Pandas dataframe with ID column
            id_column: Name of the ID column
            n_jobs: Number of parallel jobs (None = cpu_count)
            preload_nodes: Whether to preload nodes (recommended True)

        Returns:
            Dataframe with new 'skills' and 'nodes' columns
        """
        if n_jobs is None:
            n_jobs = cpu_count()

        logger.info(f"\n  Processing Configuration:")
        logger.info(f"   Dataframe rows: {len(df):,}")
        logger.info(f"   Parallel workers: {n_jobs}")
        logger.info(f"   Strategy: '{self.loading_strategy}'")

        # Pre-load nodes if using full or batch strategy
        if preload_nodes and self.loading_strategy != 'streaming':
            required_ids = set(df[id_column].astype(str).unique())
            logger.info(f"\n Pre-loading nodes...")
            self.load_nodes(required_ids)
            logger.info(f" Pre-loaded in {self.load_time:.2f}s")

        # Prepare arguments for each worker
        logger.info(f"\n Preparing processing tasks...")
        args_list = [
            (idx, id_val, self.skills_dict, self.nodes_dict or {})
            for idx, id_val in zip(df.index, df[id_column])
        ]

        # Process in parallel with optimized chunking
        logger.info(f" Starting parallel processing...")
        process_start = time.time()
        results = []
        
        optimal_chunksize = max(1, len(args_list) // (n_jobs * 4))
        
        with Pool(processes=n_jobs) as pool:
            with tqdm(total=len(args_list), desc="Processing", unit="row") as pbar:
                for result in pool.imap_unordered(
                    process_row_optimized, 
                    args_list,
                    chunksize=optimal_chunksize
                ):
                    results.append(result)
                    pbar.update(1)

        process_time = time.time() - process_start
        
        # Unpack results
        logger.info(f" Processing completed in {process_time:.2f}s")
        indices, skills_list, nodes_list = zip(*results)

        # Add to dataframe efficiently (pre-allocated)
        logger.info(f" Updating dataframe...")
        result_df = df.copy()
        result_df['skills'] = [None] * len(df)
        result_df['nodes'] = [None] * len(df)

        for idx, skills, nodes in zip(indices, skills_list, nodes_list):
            result_df.at[idx, 'skills'] = skills
            result_df.at[idx, 'nodes'] = nodes

        logger.info(f" Updated {len(result_df):,} rows successfully")
        
        # Print timing summary
        self._print_timing_summary(len(df), process_time)
        
        return result_df

    def _print_timing_summary(self, num_rows: int, process_time: float):
        """Print summary of timing information."""
        total_time = (self.load_time or 0) + process_time
        time_per_row = (process_time / num_rows) * 1000  # ms per row
        
        logger.info(f"\n  TIMING SUMMARY:")
        logger.info(f"   Load time: {self.load_time:.2f}s" if self.load_time else "   Load time: N/A (streaming)")
        logger.info(f"   Process time: {process_time:.2f}s")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Per-row: {time_per_row:.2f}ms")
        logger.info(f"   Rows/sec: {num_rows / process_time:.1f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(model_name, path):
    """Main execution function with all optimizations."""
    
    logger.info("="*70)
    logger.info("OPTIMIZED SKILL NODE PROCESSOR")
    logger.info("="*70)
    
    # Configuration
    
    
    # ========== STEP 1: Check system resources ==========
    logger.info("\n STEP 1: System Resources Check")
    available_ram_gb, cpu_cores = check_system_resources()
    
    # ========== STEP 2: Load skills dictionary ==========
    logger.info("\n STEP 2: Loading Skills Dictionary")
    logger.info("Loading job-skills mapping...")
    new_path = Path(path).cwd().parents[0]
    new_path = str(new_path)+'/code/CAG_KG_map/'
    with open(f'{new_path}position_occupation_map.json', 'r') as f:
        job_skills_data = json.load(f)
    logger.info(f" Loaded {len(job_skills_data):,} job-skill mappings")
    
    # ========== STEP 3: Analyze nodes JSON file ==========
    logger.info("\n STEP 3: Analyzing Nodes JSON File")
    nodes_json_path = f'{path}/skill_all_nodes_{model_name.split("/")[-1]}.json'
    estimated_ram_gb, file_size_gb = estimate_memory_usage(nodes_json_path)
    
    # ========== STEP 4: Recommend and select strategy ==========
    logger.info("\n STEP 4: Strategy Selection")
    recommended_strategy = recommend_strategy(file_size_gb, available_ram_gb)
    
    # You can override recommendation if needed
    STRATEGY = recommended_strategy
    # STRATEGY = 'batch'  # Uncomment to force specific strategy
    
    logger.info(f" Using strategy: '{STRATEGY}'")
    
    # ========== STEP 5: Load dataframe ==========
    logger.info("\n STEP 5: Loading Dataframe")
    logger.info("Loading dataframe...")
    df = pickle.load(open(f'{path}/df_direct_freq_mul_{model_name.split("/")[-1]}.pkl', 'rb'))
    logger.info(f" Loaded {len(df):,} rows")
    
    N_JOBS = None  # Uses all available CPU cores
    
    # ========== STEP 6: Process dataframe ==========
    logger.info("\n STEP 6: Processing Dataframe")
    
    processor = SkillNodeProcessor(
        skills_dict=job_skills_data,
        nodes_json_path=nodes_json_path,
        loading_strategy=STRATEGY
    )
    
    df_result = processor.process_dataframe_parallel(
        df=df,
        id_column='id',
        n_jobs=N_JOBS,
        preload_nodes=True
    )
    
    # ========== STEP 7: Save results ==========
    logger.info("\n STEP 7: Saving Results")
    logger.info("Saving results...")
    output_path = f'{path}/df_direct_all_nodes_skills.pkl'
    df_result.to_pickle(output_path)
    logger.info(f"Saved to {output_path}")
    
    # ========== STEP 8: Display sample ==========
    logger.info("\n STEP 8: Sample Results")
    logger.info("Sample output (first 5 rows):")
    print("\n", df_result[['id', 'skills', 'nodes']].head())
    
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE!")
    logger.info("="*70)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating dataframe with the skill-node mapped for each instance.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", choices=['Qwen/Qwen2.5-0.5B-Instruct','meta-llama/Llama-3.2-1B-Instruct','google/gemma-3-270m-it','mistralai/Mistral-7B-Instruct-v0.2'], help="Model name to use for inference.", required=True)
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    args = parser.parse_args()

    model_name = args.model_name
    path = args.path
    main(model_name, path)