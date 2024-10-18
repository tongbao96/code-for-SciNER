import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_training_examples(train_file_path):
    training_examples = []
    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data["input"].startswith("Please list all scientific entities of type"):
                text = data["input"].split("\nEntities of type")[0].strip()
                target = data["target"]
                training_examples.append({"text": text, "target": target})
    return training_examples

def load_test_examples(test_file_path):
    test_data = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_data.append(data)
    return test_data

def select_demonstrations(test_text, E_candidate, model, k=5, alpha=0.5, beta=0.3, gamma=0.2):
    test_embedding = model.encode([test_text], convert_to_tensor=True)
    candidate_embeddings = model.encode(E_candidate, convert_to_tensor=True)
    cosine_similarities = cosine_similarity(test_embedding.cpu().numpy(), candidate_embeddings.cpu().numpy()).flatten()

    diversity_scores = np.abs(np.array([len(test_text) - len(cand) for cand in E_candidate]))
    sim_scores_norm = (cosine_similarities - np.min(cosine_similarities)) / (
                np.max(cosine_similarities) - np.min(cosine_similarities))
    div_scores_norm = (diversity_scores - np.min(diversity_scores)) / (
                np.max(diversity_scores) - np.min(diversity_scores))

    final_scores = alpha * sim_scores_norm + beta * div_scores_norm + gamma * (1 - div_scores_norm)
    top_k_indices = np.argsort(final_scores)[-k:]
    top_k_examples = [E_candidate[i] for i in top_k_indices]

    return top_k_examples

def get_random_candidates(E_candidate, num_examples=100,seed=42):
    random.seed(seed)
    if len(E_candidate) > num_examples:
        return random.sample(E_candidate, num_examples)
    else:
        return E_candidate

def format_example_output(example):
    formatted_output = "Text:" + example["text"].split("\nText:")[1] + "\n"
    formatted_output += "output: " + example["target"] + "\n"
    return formatted_output

def generate_full_prompt(test_example, train_examples):
    instruction_part = test_example["input"].split("\nText:")[0]
    text_part = "Text:" + test_example["input"].split("\nText:")[1]
    prompt = instruction_part + "\nHere are some examples:\n"
    for example in train_examples:
        prompt += format_example_output(example) + "\n"
    prompt += text_part + "\noutput:"
    return prompt

train_file_path = "../Dataset/SciERC/train_SciBERT_0.70_cls.json"
test_file_path = "../SciERC/test_SciBERT_0.70_cls.json"

E_candidate_full = load_training_examples(train_file_path)
test_data = load_test_examples(test_file_path)
recall_model = SentenceTransformer("./Scibert_tmp/model", device="cuda")

k = 20
for test_example in test_data:
    test_text = test_example["input"]
    E_candidate = get_random_candidates(E_candidate_full, num_examples=100)
    top_k_examples = select_demonstrations(test_text, E_candidate, recall_model, k=k)

    full_prompt = generate_full_prompt(test_example, top_k_examples)
