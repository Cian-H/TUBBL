
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="Printing analysis results")
    parser.add_argument("--model", type=str, help="Model", required=True)
    parser.add_argument("--path", type=str, default='.', help="Path to pre-saved activations.", required=False)
    parser.add_argument("--weighted", type=int, default=0, help="Whether to use weighted activations.", required=False)
    parser.add_argument("--k", type=int, default=10, help="Number of top nodes to select.", required=False)
    args = parser.parse_args()
    model_name = args.model
    path = args.path
    weighted = args.weighted
    k = args.k

    if weighted==0:
        df = pkl.load(open(f'{path}/skill_token_similarity_results_string_{model_name.split("/")[-1]}.pkl', 'rb'))

        new_df = df.groupby('job_title').agg({'above_threshold_percentage': ['mean', 'std',]}).reset_index()

        new_df['above_threshold_percentage']['mean'].hist(bins=100, figsize=(12, 8))
        plt.title(f'Histogram of Skill-Token Similarity Above Threshold Percentage (30%) for {model_name}')
        plt.xlabel('Similarity Percentage (%)')
        plt.ylabel('# Job Titles')
        plt.savefig(f'{path}/skill_token_similarity_histogram_{model_name.split("/")[-1]}.png')

        plt.clf()
        result = pkl.load(open(f'{path}/sentence_encoder_skill_matching_results_{model_name.split("/")[-1]}.pkl', 'rb'))

        df_result = pd.DataFrame.from_dict(result).transpose()

        df_result['%_matched_skills'].hist(bins=100, figsize=(12, 8))
        plt.title(f'Histogram of Skill-Output Similarity Above Threshold Percentage (30%) for {model_name}')
        plt.xlabel('Similarity Percentage (%)')
        plt.ylabel('# Job Titles')
        plt.savefig(f'{path}/skill_output_similarity_histogram_{model_name.split("/")[-1]}.png')
    else:
        result = pkl.load(open(f'{path}/sentence_encoder_skill_matching_results_{model_name.split("/")[-1]}_top{k}_weighted.pkl', 'rb'))

        df_result = pd.DataFrame.from_dict(result).transpose()

        df_result['%_matched_skills'].hist(bins=100, figsize=(12, 8))
        plt.title(f'Histogram of Skill-Output Similarity Above Threshold Percentage (30%) for {model_name} - top{k} skill nodes')
        plt.xlabel('Similarity Percentage (%)')
        plt.ylabel('# Job Titles')
        plt.savefig(f'{path}/skill_output_similarity_histogram_{model_name.split("/")[-1]}_top{k}_weighted.png')
