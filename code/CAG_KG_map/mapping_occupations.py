from rdflib import Graph, URIRef, RDF, Literal
from sentence_transformers import SentenceTransformer, util
import pickle
import json
import argparse
from tqdm import tqdm

def get_esco_occupations(nt_file):
    """Extract ESCO occupations with their prefLabel and altLabel from .nt file"""
    print(f"Loading ESCO occupations from {nt_file}...")
    OCCUPATION = URIRef("http://data.europa.eu/esco/model#Occupation")
    PREFLABEL = URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
    ALTLABEL = URIRef("http://www.w3.org/2004/02/skos/core#altLabel")
    
    g = Graph()
    g.parse(nt_file, format='nt')
    
    occupations = {}
    
    for s, _, _ in tqdm(g.triples((None, RDF.type, OCCUPATION))):
        occupation_uri = str(s)
        labels = []
        
        # Get prefLabel
        for _, _, label in g.triples((s, PREFLABEL, None)):
            if isinstance(label, Literal):
                labels.append(str(label))
        
        # Get altLabel
        for _, _, label in g.triples((s, ALTLABEL, None)):
            if isinstance(label, Literal):
                labels.append(str(label))
        
        if labels:
            occupations[occupation_uri] = labels
    
    return occupations, g

def get_occupation_skills(graph, occupation_uri):
    #print("Getting skills for occupation: ", occupation_uri)
    """Alternative method to get skills - tries different ESCO properties"""
    SKILL = URIRef("http://data.europa.eu/esco/model#Skill")
    PREFLABEL = URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
    
    # Try different possible relationships
    possible_relations = [
        URIRef("http://data.europa.eu/esco/model#hasEssentialSkill"),
        URIRef("http://data.europa.eu/esco/model#isEssentialSkillFor"),
        URIRef("http://data.europa.eu/esco/model#hasOptionalSkill"),
        URIRef("http://data.europa.eu/esco/model#isOptionalSkillFor"),
        URIRef("http://data.europa.eu/esco/model#hasAssociation"),
        URIRef("http://data.europa.eu/esco/model#isAssociationFor")
    ]
    
    skills = []
    occupation_node = URIRef(occupation_uri)
    
    for relation in possible_relations:
        for _, _, skill in graph.triples((occupation_node, relation, None)):
            # Check if it's a skill
            if (skill, RDF.type, SKILL) in graph:
                for _, _, skill_label in graph.triples((skill, PREFLABEL, None)):
                    if isinstance(skill_label, Literal):
                        skills.append(str(skill_label))
    
    return list(set(skills))

def load_position_dictionary(pickle_file):
    """Load dictionary from pickle file"""
    print(f"Loading position dictionary from {pickle_file}...")
    return pickle.load(open(pickle_file, 'rb'))

def match_occupations_to_positions(nt_file, pickle_file, model_name):
    """Main function to match ESCO occupations with dictionary positions"""
    
    # Load data
    occupations, graph = get_esco_occupations(nt_file)
    position_dict = load_position_dictionary(pickle_file)
    
    print(f"Found {len(occupations)} ESCO occupations")
    print(f"Found {len(position_dict)} positions in dictionary")
    
    # Initialize model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Prepare occupation labels for encoding (flatten all labels)
    occupation_labels = []
    occupation_uris = []
    for uri, labels in occupations.items():
        for label in labels:
            occupation_labels.append(label)
            occupation_uris.append(uri)
    
    # Prepare position labels for encoding
    position_keys = []
    position_labels = []
    for key, data in position_dict.items():
        if 'position' in data:
            position_keys.append(key)
            position_labels.append(data['position'])
    
    print(f"Encoding {len(occupation_labels)} occupation labels...")
    print(f"Encoding {len(position_labels)} position labels...")
    
    # Encode labels
    occupation_embeddings = model.encode(occupation_labels, convert_to_tensor=True)
    position_embeddings = model.encode(position_labels, convert_to_tensor=True)
    
    # Compute cosine similarities: positions x occupations
    cos_sims = util.cos_sim(position_embeddings, occupation_embeddings)
    
    results = {}
    
    for pos_idx, position_key in tqdm(enumerate(position_keys)):
        # Find best matching occupation
        best_occ_idx = cos_sims[pos_idx].argmax().item()
        similarity_score = cos_sims[pos_idx][best_occ_idx].item()
        
        best_occupation_uri = occupation_uris[best_occ_idx]
        best_occupation_label = occupation_labels[best_occ_idx]
        
        # Get skills for this occupation
        skills = get_occupation_skills(graph, best_occupation_uri)
        
        results[position_key] = {
            "position": position_dict[position_key]['position'],
            "occupation": best_occupation_label,
            "occupation_uri": best_occupation_uri,
            "cosine_similarity": similarity_score,
            "skills": skills
        }
    
    # Save results
    with open(f'position_occupation_map.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to position_occupation_map")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mapping occupation script.")
    parser.add_argument("--ontology", type=str, help="Ontology file", required=True)
    parser.add_argument("--data", type=str, help="Pickle dictionary", required=True)
    parser.add_argument("--model", type=str, help="Model", required=True)
    
    args = parser.parse_args()

    nt_file = args.ontology
    pickle_file = args.data
    model_name = args.model
    results = match_occupations_to_positions(nt_file, pickle_file, model_name)
    print(f"Processed {len(results)} position-occupation matches")
   