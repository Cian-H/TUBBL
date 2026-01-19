from rdflib import Graph, URIRef, RDF, Literal
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import json
import argparse

# 1. Extract ESCO skills (prefLabel) from .nt file
def get_esco_skills(nt_file):
    SKILL = URIRef("http://data.europa.eu/esco/model#Skill")
    PREFLABEL = URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
    g = Graph()
    g.parse(nt_file, format='nt')
    skills = {}
    for s, _, _ in g.triples((None, RDF.type, SKILL)):
        for _, _, label in g.triples((s, PREFLABEL, None)):
            if isinstance(label, Literal):
                skills[str(s)] = str(label)
    return skills

# 2. Get networkx node labels
def networkx_node_labels(nxgraph):
    labels = {}
    for node, datadict in nxgraph.nodes(data=True):
        labels[node] = datadict.get('token', str(node))
    return labels

# 3. Main function for skill matching
def make_skill_matching(nt_file, g_nx, out_json='node_skill_mapping.json'):
    skills = get_esco_skills(nt_file)
    nx_labels = networkx_node_labels(g_nx)
    
    model = SentenceTransformer('all-mpnet-base-v2')
    
    skill_labels_list = list(skills.values())
    node_labels_list = list(nx_labels.values())
    
    # Encode texts
    skill_embeddings = model.encode(skill_labels_list, convert_to_tensor=True)
    node_embeddings = model.encode(node_labels_list, convert_to_tensor=True)
    
    # Compute cosine similarities
    cos_sims = util.cos_sim(node_embeddings, skill_embeddings)
    
    matches = {}
    for idx, node in enumerate(nx_labels):
        best_skill_idx = cos_sims[idx].argmax().item()
        similarity_score = cos_sims[idx][best_skill_idx].item()
        
        matches[node] = {
            "node_label": nx_labels[node],
            "skill_uri": list(skills.keys())[best_skill_idx],
            "skill_label": skill_labels_list[best_skill_idx],
            "cosine_similarity": similarity_score
        }
    
    # Save results
    with open(out_json, 'w') as f:
        json.dump(matches, f, indent=2)
    
    return matches

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run mapping skills script.")
    parser.add_argument("--graph", type=str, help="Graphml file", required=True)
    parser.add_argument("--ontology", type=str, help="Ontology file", required=True)
    parser.add_argument("--model", type=str, help="Model", required=True)
    
    args = parser.parse_args()
    G = nx.read_graphml(args.graph)
    nt_file = args.ontology
    result = make_skill_matching(nt_file, G, f'node_skill_mapping_{args.model}.json')
