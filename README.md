# Towards Unveiling the Black Box of LLMs: Understanding Skill Inference through Co-Activation and Knowledge Graphs

## General Aspects
The codes presented here were developed for the XAI'26 conference paper. To execute the methods, a Huggingfaces account is required in order to access the models and the job datasets. It is recommend use the conda enviroment presented here: https://kgtk.readthedocs.io/en/latest/install/ and install the requirements.txt.

The repository is separated in 3 parts:
- Coactivation Graph - Knowledge Graph mapping: The mapping scripts between the data and the knowledge graph
- Inference Coactivation: Model's output and coactivation graph development
- Taxo Onto: Process to build the knowledge graph from a taxonomy

## Reproducibility
The structure for the reproducibility is:
- Download the O*NET database - https://www.onetcenter.org/dl_files/database/db_30_1_mysql.zip
- Set the Database configuration on 'code/tax_onto/config.py'
- Run the script 'code/tax_onto/onet_taxo_kg.py' to generate the Knowledge Graph
- Run the script 'code/inference_coactivation/inference_coactivation.py' passing the model name, the number of neurons, the number of samples, the threshold to build the coactivation graph and the saving path
- Run the script 'code/CAG_KG_map/mapping_occupations.py' to generate the mapping between the dataset and knowledge graph job titles
- The sequence of next scripts is stored in the 'experiment' folder

## Reference
```
@article{
}
```
