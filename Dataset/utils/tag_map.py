def get_keys(data_type):
    tag_map = get_tag_map(data_type)
    try:
        keys = list(set([tag_map[x][tag_map[x].index("-") + 1:].lower() for x in tag_map if x != 0]))
    except:
        keys = list(set([tag_map[x].lower() for x in tag_map if x != 0]))
    return keys

def get_tag_map(data_type):
    if data_type == "bc5cdr":
        tag_map = {
            0: "O",
            1: "B-Chemical",
            2: "B-Disease",
            3: "I-Disease",
            4: "I-Chemical"
        }

    elif data_type == "jnlpba":
        tag_map = {
            0: "O",
            1: "B-DNA",
            2: "I-DNA",
            3: "B-protein",
            4: "I-protein",
            5: "B-cell_type",
            6: "I-cell_type",
            7: "B-cell_line",
            8: "I-cell_line",
            9: "B-RNA",
            10: "I-RNA"
        }
    elif data_type == "SciERC":
            tag_map = {
            0: "O",
            1: "B-Generic",
            2: "I-Generic",
            3: "B-Material",
            4: "I-Material",
            5: "B-Method",
            6: "I-Method",
            7: "B-Metric",
            8: "I-Metric",
            9: "B-Task",
            10: "I-Task",
            11: "B-Otherscientificterm",
            12: "I-Otherscientificterm"
        }
    return tag_map

def get_entity_type_desc(data_type):
    if data_type == "bc5cdr":
        entity_type_desc = {
            "chemical": ("Chemical, refers to substances with a defined chemical composition, including elements, compounds,commonly used to describe any material with a specific molecular structure or chemical properties.", "Aspirin is commonly used to reduce fever."),
            "disease": ("Disease, refers to medical conditions characterized by specific symptoms and signs, including infections, genetic disorders, and chronic diseases.", "COVID-19 is caused by the SARS-CoV-2 virus.")
            #Here the data is just a sample; you can find examples containing the specified entity types from the train file yourself.
}
    elif data_type == "jnlpba":
        entity_type_desc = {
            "dna": ("DNA, short for deoxyribonucleic acid, is the molecule that contains the genetic code of organisms.", "The DNA sequence ATCG is common in human genomes."),
            "protein": ("Proteins are polymer chains made of amino acids linked together by peptide bonds.", "Insulin is a protein hormone used to control blood sugar levels."),
            "cell_type": ("A cell type is a classification used to identify cells that share morphological or phenotypical features.", "T-cells play a crucial role in the immune system."),
            "cell_line": ("Cell line is a general term that applies to a defined population of cells that can be maintained in culture for an extended period of time, retaining stability of certain phenotypes and functions.", "HeLa cells are an immortal cell line used in scientific research."),
            "rna": ("Ribonucleic acid (RNA) is a molecule that is present in the majority of living organisms and viruses.", "mRNA vaccines use a small piece of the SARS-CoV-2 virus's mRNA to instruct cells to produce a protein that triggers an immune response.")
          #Here the data is just a sample; you can find examples containing the specified entity types from the train file yourself.
        }

    elif data_type == "SciERC":
        entity_type_desc = {
            "Task": ( "Applications, problems to solve, systems to construct.Examples include 'information extraction', 'machine reading system', 'image segmentation'.", "Our approach is primarily used for information extraction."),
            "Method": ("Methods , models, systems to use, or tools, components of a system, frameworks.Examples include 'language model', 'CORENLP', 'POS parser'.","We use a language model to generate natural language text."),
            "Metric": (" Metrics, measures, or entities that can express quality of a system / method.Examples include 'F1', 'BLEU', 'Precision', 'time complexity'.","The model achieved an F1 score of 92% on the test dataset."),
            "Material": ("Data, datasets, resources, Corpus, Knowledge base.Examples include 'speech data', 'stereo images', 'CoNLL', 'Wikipedia'.","We used a large set of speech data to train the model."),
            "Generic": ("General terms or pronouns that may refer to a entity but are not themselves informative, often used as connection words.Examples include 'model', 'approach', 'them'.","The performance of the model exceeded that of previous approaches."),
            "OtherScientificTerm": (" Phrases that are a scientific terms but do not fall into any of the above classes.Examples include 'physical' or 'geometric constraints', 'qualitative prior knowledge','noise'.","The performance of the model is influenced by noise and qualitative prior knowledge."),
            # Here the data is just a sample; you can find examples containing the specified entity types from the train file yourself.
        }
    return entity_type_desc
