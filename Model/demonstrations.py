import argparse
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from nltk import RegexpParser, Tree, pos_tag
from nltk.tokenize import TreebankWordTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def split_input(input_text):
    instruction, text_part = input_text.split("\nText:", 1)
    note = ""

    if "\nEntities of type" in text_part:
        text, note = text_part.split("\nEntities of type", 1)
        note = "Entities of type " + note.strip()
    elif "\nNote:" in text_part:
        text, note = text_part.split("\nNote:", 1)
    else:
        text = text_part

    return instruction.strip(), text.strip(), note.strip()


def get_target(data):
    target_dict = defaultdict(list)

    for label in data.get("labels", []):
        entity_type = label.get("name", label.get("type"))
        entity = label.get("value", label.get("text"))
        if entity_type is not None and entity is not None:
            if entity not in target_dict[entity_type]:
                target_dict[entity_type].append(entity)

    if len(target_dict) == 0:
        target = data.get("target", "")
        try:
            target_json = json.loads(target)
            if isinstance(target_json, dict):
                for entity_type, entities in target_json.items():
                    if not isinstance(entities, list):
                        entities = [entities]
                    target_dict[entity_type].extend(entities)
        except:
            pairs = re.findall(r"\(\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)", target)
            for entity_type, entity in pairs:
                entity_type = entity_type.strip().strip("'\"")
                entity = entity.strip().strip("'\"")
                if entity not in target_dict[entity_type]:
                    target_dict[entity_type].append(entity)

    if len(target_dict) == 0:
        return "None", []

    target = json.dumps(dict(target_dict), ensure_ascii=False, separators=(",", ":"))
    return target, list(target_dict.keys())


def load_training_examples(train_file_path):
    training_examples = []

    with open(train_file_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            data = json.loads(line)
            if data["input"].startswith("Please list all scientific entities of type"):
                _, text, _ = split_input(data["input"])
                target, entity_types = get_target(data)
                training_examples.append({
                    "line_id": line_id,
                    "raw_index": data.get("raw_index", line_id),
                    "text": text,
                    "target": target,
                    "entity_types": entity_types,
                    "length": len(text.split())
                })

    return training_examples


def load_test_examples(test_file_path):
    test_examples = []

    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            data = json.loads(line)
            if data["input"].startswith("Please list all scientific entities of type"):
                _, text, _ = split_input(data["input"])
                data["line_id"] = line_id
                data["text"] = text
                test_examples.append(data)

    return test_examples


def get_random_candidates(training_examples, num_examples=100, seed=42,
                          lower_quantile=0.1, upper_quantile=0.9):
    lengths = np.array([example["length"] for example in training_examples])
    lower_length = np.quantile(lengths, lower_quantile)
    upper_length = np.quantile(lengths, upper_quantile)

    candidates = [
        example for example in training_examples
        if lower_length <= example["length"] <= upper_length
    ]

    target_size = min(num_examples, len(training_examples))
    if len(candidates) < target_size:
        median_length = np.median(lengths)
        selected_ids = set([example["line_id"] for example in candidates])
        rest_examples = [
            example for example in training_examples
            if example["line_id"] not in selected_ids
        ]
        rest_examples.sort(key=lambda x: abs(x["length"] - median_length))
        candidates.extend(rest_examples[:target_size - len(candidates)])

    random_generator = random.Random(seed)
    if len(candidates) > target_size:
        candidates = random_generator.sample(candidates, target_size)
    else:
        random_generator.shuffle(candidates)

    return candidates, lower_length, upper_length


def get_syntax_tree(text, chunk_parser, tokenizer):
    tokens = tokenizer.tokenize(text)
    try:
        tagged_tokens = pos_tag(tokens)
    except LookupError as error:
        raise RuntimeError(
            "NLTK POS tagger is missing. Run: "
            "python -m nltk.downloader averaged_perceptron_tagger_eng"
        ) from error

    chunk_tree = chunk_parser.parse(tagged_tokens)

    def remove_words(tree):
        if isinstance(tree, Tree):
            return Tree(tree.label(), [remove_words(child) for child in tree])
        return Tree(tree[1], [])

    return remove_words(chunk_tree)


def get_postorder_data(tree):
    nodes = [None]
    leftmost = [0]

    def visit(node):
        first_leftmost = None
        for child in node:
            child_index = visit(child)
            if first_leftmost is None:
                first_leftmost = leftmost[child_index]

        nodes.append(node)
        node_index = len(nodes) - 1
        if first_leftmost is None:
            leftmost.append(node_index)
        else:
            leftmost.append(first_leftmost)
        return node_index

    visit(tree)
    return nodes, leftmost


def get_keyroots(leftmost):
    keyroots = {}
    for i in range(1, len(leftmost)):
        keyroots[leftmost[i]] = i
    return sorted(keyroots.values())


def tree_edit_distance(tree_1, tree_2):
    nodes_1, leftmost_1 = get_postorder_data(tree_1)
    nodes_2, leftmost_2 = get_postorder_data(tree_2)

    size_1 = len(nodes_1) - 1
    size_2 = len(nodes_2) - 1
    tree_distance = [[0] * (size_2 + 1) for _ in range(size_1 + 1)]

    for root_1 in get_keyroots(leftmost_1):
        for root_2 in get_keyroots(leftmost_2):
            start_1 = leftmost_1[root_1]
            start_2 = leftmost_2[root_2]
            rows = root_1 - start_1 + 2
            columns = root_2 - start_2 + 2
            forest_distance = [[0] * columns for _ in range(rows)]

            for i in range(start_1, root_1 + 1):
                row = i - start_1 + 1
                forest_distance[row][0] = forest_distance[row - 1][0] + 1

            for j in range(start_2, root_2 + 1):
                column = j - start_2 + 1
                forest_distance[0][column] = forest_distance[0][column - 1] + 1

            for i in range(start_1, root_1 + 1):
                row = i - start_1 + 1
                for j in range(start_2, root_2 + 1):
                    column = j - start_2 + 1
                    delete_cost = forest_distance[row - 1][column] + 1
                    insert_cost = forest_distance[row][column - 1] + 1

                    if leftmost_1[i] == start_1 and leftmost_2[j] == start_2:
                        update_cost = 0 if nodes_1[i].label() == nodes_2[j].label() else 1
                        replace_cost = forest_distance[row - 1][column - 1] + update_cost
                        distance = min(delete_cost, insert_cost, replace_cost)
                        forest_distance[row][column] = distance
                        tree_distance[i][j] = distance
                    else:
                        prefix_row = leftmost_1[i] - start_1
                        prefix_column = leftmost_2[j] - start_2
                        subtree_cost = forest_distance[prefix_row][prefix_column] + tree_distance[i][j]
                        forest_distance[row][column] = min(delete_cost, insert_cost, subtree_cost)

    return tree_distance[size_1][size_2]


def get_tree_size(tree):
    return 1 + sum([get_tree_size(child) for child in tree])


def get_structural_diversity(tree_1, tree_2):
    distance = tree_edit_distance(tree_1, tree_2)
    max_tree_size = max(get_tree_size(tree_1), get_tree_size(tree_2))
    return distance / max_tree_size


def prepare_candidates(candidates, model, chunk_parser, tokenizer, entity_type_num):
    candidate_texts = [candidate["text"] for candidate in candidates]
    candidate_embeddings = model.encode(
        candidate_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    for i, candidate in enumerate(tqdm(candidates, desc="Parsing candidate syntax trees")):
        candidate["embedding"] = candidate_embeddings[i]
        candidate["type_diversity"] = len(set(candidate["entity_types"])) / entity_type_num
        candidate["syntax_tree"] = get_syntax_tree(candidate["text"], chunk_parser, tokenizer)

    return candidates


def select_demonstrations(test_text, candidates, model, chunk_parser, tokenizer,
                          k=20, alpha=0.4, beta=0.4, gamma=0.2):
    test_embedding = model.encode(
        [test_text],
        convert_to_tensor=True,
        normalize_embeddings=True
    )[0]
    candidate_embeddings = torch.stack([candidate["embedding"] for candidate in candidates])
    similarity_scores = F.cosine_similarity(
        test_embedding.unsqueeze(0), candidate_embeddings, dim=1
    )
    test_tree = get_syntax_tree(test_text, chunk_parser, tokenizer)

    results = []
    for i, candidate in enumerate(candidates):
        similarity = similarity_scores[i].item()
        type_diversity = candidate["type_diversity"]
        structural_diversity = get_structural_diversity(
            test_tree, candidate["syntax_tree"]
        )
        score = (
            alpha * similarity
            + beta * type_diversity
            + gamma * structural_diversity
        )
        results.append({
            "candidate": candidate,
            "similarity": similarity,
            "type_diversity": type_diversity,
            "structural_diversity": structural_diversity,
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:min(k, len(results))]


def format_example_output(example):
    output = "Text: " + example["text"] + "\n"
    output += "Output: " + example["target"]
    return output


def generate_full_prompt(test_example, selected_examples):
    instruction, test_text, note = split_input(test_example["input"])
    prompt = instruction + "\nHere are some examples:\n"

    for result in selected_examples:
        prompt += format_example_output(result["candidate"]) + "\n"

    prompt += "Text: " + test_text + "\n"
    if note:
        prompt += "Note: " + note + "\n"
    prompt += "Output:"
    return prompt


def save_prompts(test_examples, candidates, model, chunk_parser, tokenizer,
                 output_file_path, k=20, alpha=0.4, beta=0.4, gamma=0.2):
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for test_example in tqdm(test_examples, desc="Selecting demonstrations"):
            selected_examples = select_demonstrations(
                test_example["text"],
                candidates,
                model,
                chunk_parser,
                tokenizer,
                k=k,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )

            selected_results = []
            for result in selected_examples:
                candidate = result["candidate"]
                selected_results.append({
                    "line_id": candidate["line_id"],
                    "raw_index": candidate["raw_index"],
                    "text": candidate["text"],
                    "target": candidate["target"],
                    "entity_types": candidate["entity_types"],
                    "similarity": result["similarity"],
                    "type_diversity": result["type_diversity"],
                    "structural_diversity": result["structural_diversity"],
                    "score": result["score"]
                })

            output = {
                "line_id": test_example["line_id"],
                "raw_index": test_example.get("raw_index", test_example["line_id"]),
                "input": test_example["input"],
                "prompt": generate_full_prompt(test_example, selected_examples),
                "selected_examples": selected_results
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()


TRAIN_FILE_PATH = "data/SciERC/train_SciBERT_0.70_cls.json"
TEST_FILE_PATH = "data/SciERC/test_SciBERT_0.70_cls.json"
MODEL_PATH = "./Scibert-tmp/model"
OUTPUT_FILE_PATH = "outputs/SciERC/paper_aligned_prompts.jsonl"

CANDIDATE_NUM = 100
K = 20
ALPHA = 0.4
BETA = 0.4
GAMMA = 0.2
SEED = 42
LOWER_QUANTILE = 0.1
UPPER_QUANTILE = 0.9

CHUNK_GRAMMAR = r"""
    NP: {<DT|PRP\$>?<JJ.*>*<NN.*|PRP>+}
    PP: {<IN><NP>}
    ADJP: {<RB.*>*<JJ.*>+}
    ADVP: {<RB.*>+}
    VP: {<MD>?<VB.*><NP|PP|ADJP|ADVP>*}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default=TRAIN_FILE_PATH)
    parser.add_argument("--test_file_path", type=str, default=TEST_FILE_PATH)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--output_file_path", type=str, default=OUTPUT_FILE_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--candidate_num", type=int, default=CANDIDATE_NUM)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    training_examples = load_training_examples(args.train_file_path)
    test_examples = load_test_examples(args.test_file_path)
    candidates, lower_length, upper_length = get_random_candidates(
        training_examples,
        num_examples=args.candidate_num,
        seed=args.seed,
        lower_quantile=LOWER_QUANTILE,
        upper_quantile=UPPER_QUANTILE
    )

    entity_types = set()
    for example in training_examples:
        entity_types.update(example["entity_types"])

    print("Training examples: ", len(training_examples))
    print("Test examples: ", len(test_examples))
    print("Candidate examples: ", len(candidates))
    print("Candidate sentence length: ", lower_length, "-", upper_length)

    model = SentenceTransformer(args.model_path, device=args.device)
    chunk_parser = RegexpParser(CHUNK_GRAMMAR)
    tokenizer = TreebankWordTokenizer()
    candidates = prepare_candidates(
        candidates,
        model,
        chunk_parser,
        tokenizer,
        max(1, len(entity_types))
    )

    save_prompts(
        test_examples,
        candidates,
        model,
        chunk_parser,
        tokenizer,
        args.output_file_path,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    print("Output written to: ", args.output_file_path)
