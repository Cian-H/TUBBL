import csv
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, SKOS, URIRef
from tqdm import tqdm

import config as cfg

import mysql.connector
import pandas as pd

def extract_and_transform():
    # Database connection details
    db_config = {
        'host': cfg.MYSQL_CONFIG['host'],
        'username': cfg.MYSQL_CONFIG['username'],
        'password': cfg.MYSQL_CONFIG['password'],
        'database': cfg.MYSQL_CONFIG['database']
    }
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        triples = []

        # 1. Extract Occupations
        cursor.execute("SELECT onetsoc_code, title FROM occupation_data")
        occupations = cursor.fetchall()
        for code, title in occupations:
            # ESCO Occupation concept
            occupation_uri = f"{cfg.ONET_IRI}{code.replace(' ', '_')}"
            triples.append((occupation_uri, str(RDF.type), str(cfg.ESCO_OCCUPATION_CLASS)))
            triples.append((occupation_uri, str(SKOS.prefLabel), Literal(title, lang="en")))

        # 1.2 Extract Alternate Titles
        cursor.execute("SELECT onetsoc_code, alternate_title FROM alternate_titles")
        alternate_titles = cursor.fetchall()
        for onetsoc_code, alternate_title in alternate_titles:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            triples.append((occupation_uri, str(SKOS.altLabel), Literal(alternate_title, lang="en")))

        # 1.3 Extract related occupations
        cursor.execute("SELECT onetsoc_code, related_onetsoc_code FROM related_occupations")
        related_occupations = cursor.fetchall()
        for onetsoc_code, related_onetsoc_code in related_occupations:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            related_occupation_uri = f"{cfg.ONET_IRI}{related_onetsoc_code.replace(' ', '_')}"
            
            triples.append((occupation_uri, str(cfg.ESCO_HAS_ASSOCIATION), related_occupation_uri))
            triples.append((related_occupation_uri, str(cfg.ESCO_IS_ASSOCIATON_FOR), occupation_uri))

        # 2. Extract Skills related to Occupations
        cursor.execute("SELECT onetsoc_code, S.element_id, scale_id, data_value, element_name FROM skills S INNER JOIN content_model_reference CS on S.element_id = CS.element_id AND S.scale_id = 'IM'")
        occupation_skills = cursor.fetchall()
        for onetsoc_code, element_id, scale_id, data_value, element_name in occupation_skills:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            skill_uri = f"{cfg.ONET_IRI}Skill/{element_id.replace(' ', '_')}"
            
            triples.append((skill_uri, str(RDF.type), str(cfg.ESCO_SKILL_CLASS)))
            triples.append((skill_uri, str(SKOS.prefLabel), Literal(element_name, lang="en")))

            # Determine relationship based on importance (threshold = 3)
            if data_value >= 3:
                relationship = str(cfg.ESCO_HAS_ESSENTIAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_ESSENTIAL_SKILL_FOR)
            else:
                relationship = str(cfg.ESCO_HAS_OPTIONAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_OPTIONAL_SKILL_FOR)
            
            # Add occupation-skill relationships
            triples.append((occupation_uri, relationship, skill_uri))
            triples.append((skill_uri, inverse_rel, occupation_uri))

        # 2.1 Extract Technical Skills
        cursor.execute("SELECT onetsoc_code, example FROM technology_skills")
        tech_skills = cursor.fetchall()
        for onetsoc_code, example in tech_skills:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            tech_skill_uri = f"{cfg.ONET_IRI}TechnologySkill/{example.replace(' ', '_')}"
            
            triples.append((tech_skill_uri, str(RDF.type), str(cfg.ESCO_SKILL_CLASS)))
            triples.append((tech_skill_uri, str(cfg.ESCO_SKILL_TYPE), Literal("technology_skills", lang="en")))
            triples.append((tech_skill_uri, str(SKOS.prefLabel), Literal(example, lang="en")))
            
            # Relationship: Occupation hasTechnologySkill TechnologySkill
            triples.append((occupation_uri, str(cfg.ESCO_HAS_ASSOCIATION), tech_skill_uri))
            triples.append((tech_skill_uri, str(cfg.ESCO_IS_ASSOCIATON_FOR), occupation_uri))

        # 2.2 Extract Knowledge related to Occupations
        cursor.execute("SELECT onetsoc_code, S.element_id, scale_id, data_value, element_name FROM knowledge S INNER JOIN content_model_reference CS on S.element_id = CS.element_id AND S.scale_id = 'IM'")
        occupation_knowledge = cursor.fetchall()
        for onetsoc_code, element_id, scale_id, data_value, element_name in occupation_knowledge:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            knowledge_uri = f"{cfg.ONET_IRI}Knowledge/{element_id.replace(' ', '_')}"
            triples.append((knowledge_uri, str(RDF.type), str(cfg.ESCO_SKILL_CLASS)))
            triples.append((knowledge_uri, str(SKOS.prefLabel), Literal(element_name, lang="en")))
            triples.append((knowledge_uri, str(cfg.ESCO_SKILL_TYPE), Literal("knowledge", lang="en")))
            # Determine relationship based on importance (threshold = 3)
            if data_value >= 3:
                relationship = str(cfg.ESCO_HAS_ESSENTIAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_ESSENTIAL_SKILL_FOR)
            else:
                relationship = str(cfg.ESCO_HAS_OPTIONAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_OPTIONAL_SKILL_FOR) 
            
            # Add occupation-knowledge relationships
            triples.append((occupation_uri, relationship, knowledge_uri))
            triples.append((knowledge_uri, inverse_rel, occupation_uri))

        # 2.3 Extract Work Activities related to Occupations
        cursor.execute("SELECT onetsoc_code, S.element_id, scale_id, data_value, element_name FROM work_activities S INNER JOIN content_model_reference CS on S.element_id = CS.element_id AND S.scale_id = 'IM'")
        occupation_work_activities = cursor.fetchall()
        for onetsoc_code, element_id, scale_id, data_value, element_name in occupation_work_activities:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            work_activity_uri = f"{cfg.ONET_IRI}WorkActivity/{element_id.replace(' ', '_')}"
            triples.append((work_activity_uri, str(RDF.type), str(cfg.ESCO_SKILL_CLASS)))
            triples.append((work_activity_uri, str(SKOS.prefLabel), Literal(element_name, lang="en")))
            triples.append((work_activity_uri, str(cfg.ESCO_SKILL_TYPE), Literal("work_activity", lang="en")))
            # Determine relationship based on importance (threshold = 3)
            if data_value >= 3:
                relationship = str(cfg.ESCO_HAS_ESSENTIAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_ESSENTIAL_SKILL_FOR)
            else:
                relationship = str(cfg.ESCO_HAS_OPTIONAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_OPTIONAL_SKILL_FOR)
            # Add occupation-work activity relationships
            triples.append((occupation_uri, relationship, work_activity_uri))
            triples.append((work_activity_uri, inverse_rel, occupation_uri))
    
        # 2.4 Extract Abilities related to Occupations
        cursor.execute("SELECT onetsoc_code, S.element_id, scale_id, data_value, element_name FROM abilities S INNER JOIN content_model_reference CS on S.element_id = CS.element_id AND S.scale_id = 'IM'")
        occupation_abilities = cursor.fetchall()
        for onetsoc_code, element_id, scale_id, data_value, element_name in occupation_abilities:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            ability_uri = f"{cfg.ONET_IRI}Ability/{element_id.replace(' ', '_')}"
            triples.append((ability_uri, str(RDF.type), str(cfg.ESCO_SKILL_CLASS)))
            triples.append((ability_uri, str(SKOS.prefLabel), Literal(element_name, lang="en")))
            triples.append((ability_uri, str(cfg.ESCO_SKILL_TYPE), Literal("ability", lang="en")))
            # Determine relationship based on importance (threshold = 3)
            if data_value >= 3:
                relationship = str(cfg.ESCO_HAS_ESSENTIAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_ESSENTIAL_SKILL_FOR)
            else:
                relationship = str(cfg.ESCO_HAS_OPTIONAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_OPTIONAL_SKILL_FOR)
            # Add occupation-ability relationships
            triples.append((occupation_uri, relationship, ability_uri))
            triples.append((ability_uri, inverse_rel, occupation_uri))

        # 3. Extract Education related to Occupations
        # Using 'education_training_experience' table with 'element_id' and 'data_value'
        cursor.execute("SELECT onetsoc_code, S.element_id, scale_id, data_value, element_name FROM education_training_experience S INNER JOIN content_model_reference CS on S.element_id = CS.element_id AND S.scale_id = 'IM'")
        occupation_education = cursor.fetchall()
        for onetsoc_code, element_id, scale_id, data_value, element_name in occupation_education:
            occupation_uri = f"{cfg.ONET_IRI}{onetsoc_code.replace(' ', '_')}"
            education_uri = f"{cfg.ONET_IRI}Education/{element_id.replace(' ', '_')}"

            triples.append((education_uri, str(RDF.type), str(cfg.ESCO_SKILL_CLASS)))
            triples.append((education_uri, str(SKOS.prefLabel), Literal(element_name, lang="en")))
            triples.append((education_uri, str(cfg.ESCO_SKILL_TYPE), Literal("education", lang="en")))
            # Determine relationship based on importance (threshold = 3)
            if data_value >= 3:
                relationship = str(cfg.ESCO_HAS_ESSENTIAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_ESSENTIAL_SKILL_FOR)
            else:
                relationship = str(cfg.ESCO_HAS_OPTIONAL_SKILL)
                inverse_rel = str(cfg.ESCO_IS_OPTIONAL_SKILL_FOR)
            # Add occupation-education relationships
            triples.append((occupation_uri, relationship, education_uri))
            triples.append((education_uri, inverse_rel, occupation_uri))

        # Create DataFrame and save to CSV
        df = pd.DataFrame(triples, columns=['subject', 'predicate', 'object'])
        # For TSV output
        df.to_csv('onet_esco_knowledge_graph.tsv', index=False)
        # Format N_triples 
        df['subject'] = df['subject'].apply(lambda x: '<' + x + '>' if x.startswith('http') else f"{x}")
        df['predicate'] = df['predicate'].apply(lambda x: '<' + x + '>' if x.startswith('http') else f"{x}")
        df['object'] = df['object'].apply(lambda x: '<' + x + '>' if x.startswith('http') else f'"{x}"')
        df['end'] = '.'
        df.to_csv('onet_esco_knowledge_graph.nt', index=False, header=False, sep=' ', quoting=csv.QUOTE_NONE, escapechar=' ')
        
        print("Knowledge graph successfully extracted and saved to onet_esco_knowledge_graph.csv")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    extract_and_transform()