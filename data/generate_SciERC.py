import json
import random
import os
import copy
import torch
import pickle

import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from utils.nest import get_key_map, get_keys, get_entity_type_desc
from collections import defaultdict



def get_topk(model, input, entity_info, topk=5, threshold=0.0, w1=0.5, w2=0.5):
    descriptions = [info[0] for info in entity_info]
    sentences = [info[1] for info in entity_info]
    entity_types = [x for x in get_entity_type_desc(data_type).keys()]

    embeddings = model.encode([input] + descriptions + sentences, convert_to_tensor=True, normalize_embeddings=True)
    input_embedding = embeddings[0]
    desc_embeddings = embeddings[1:len(descriptions) + 1]
    sent_embeddings = embeddings[len(descriptions) + 1:]

    desc_similarities = F.cosine_similarity(input_embedding.unsqueeze(0), desc_embeddings)
    sent_similarities = F.cosine_similarity(input_embedding.unsqueeze(0), sent_embeddings)

    combined_similarities = w1 * desc_similarities + w2 * sent_similarities
    topk_scores, topk_indices = torch.topk(combined_similarities, topk)

    topk_results = [
        entity_types[topk_indices[i]]
        for i in range(len(topk_indices))
        if topk_scores[i] >= threshold
    ]
    return topk_results, topk_scores[:len(topk_results)]

def add_sample_id(raw_json, raw_index):
    raw_json["raw_index"] = raw_index
    return raw_json

def ner_t5_data_format(fout, item, schemas, mode='train'):
    global recall_model
    entity_info = [(desc, sentence) for x in schemas for desc, sentence in [get_entity_type_desc(data_type)[x.lower()]]]
    query, score = get_topk(recall_model, item["text"], entity_info, len(entity_info), 0.9, w1=0.5, w2=0.5)
    # query = [x.split(": ")[0] for x in query]
    input = "Please list all scientific entities of type [{}] in the following text. Your output should follow the JSON format: {{Type: [entities]}}. If no entities, return None. \nText: {}\nEntities of type [{}] may exist in text".format(
        ", ".join(schemas), " ".join(item["text"]), ", ".join(query))
    target = []
    for label in item["label"]:
        target.append("({}, {})".format(label["name"], label["value"]))

    target = ", ".join(target)
    target = "[" + target + "]"
    item = {
        "input": input,
        "target": target,
        "labels": item["label"],
        "raw_index": item["raw_index"]
    }
    fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def auxiliary_t5_data_format(fout, item, schemas):
    entity_words = [label["value"] for label in item["label"]]
    # input = "List all entity types in the text of the type [{}].\nText: {}".format(", ".join(schemas), item["text"])
    input = 'Text:{}\nPlease categorize these scientific entities [{}] from the text. Type options are [{}]. Your output should follow the JSON format: {{Type:Entities}}.'.format(
        " ".join(item["text"]),
        ', '.join(entity_words),
        ', '.join(schemas)
    )
    target = []
    for label in item["label"]:
        # target.append("{}".format(label["name"]))
        target.append("({}, {})".format(label["name"], label["value"]))
    target = list(set(target))
    target = ", ".join(target)
    target = "[" + target + "]"

    item = {
        "input": input,
        "target": target,
        "labels": item["label"],
        "raw_index": item["raw_index"]
    }
    fout.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_2_t5_type(data_path):
    sample_id = 0

    data_split = ["train", 'validation', 'test']
    out_files = ["train_SciBERT_0.90_cls.json", 'valid_SciBERT_0.90_cls.json', 'test_SciBERT_0.90_cls.json']
    schemas = get_keys(data_type)
    dataset = load_dataset(data_path)

    for split, fn_out in zip(data_split, out_files):
        split_dataset = dataset[split]
        file_name = os.path.join(data_path, fn_out)
        f_out = open(file_name, 'w', encoding='utf-8')
        pbar = tqdm(total=len(split_dataset))
        for data in split_dataset:
            label = set()
            for entity in data['entities']:
                # label.add((get_key_map(data_type)[entity['type']], data['tokens'][entity['start']:entity['end']+1]))
                label.add((get_key_map(data_type)[entity['type'].lower()],
                           " ".join(data['tokens'][entity['start']:entity['end'] + 1])))

            label = [{"name": x[0], "value": x[1]} for x in label]
            item = {"text": data['tokens'], "label": label}
            item = add_sample_id(item, sample_id)
            ner_t5_data_format(fout=f_out, item=item, schemas=schemas, mode=split)
            p = random.random()
            if p < 0.5 and split == 'train':
                auxiliary_t5_data_format(f_out, item, schemas)
            sample_id += 1
            pbar.update(1)
        f_out.close()
        pbar.close()


data_type = 'SciERC'  # dataset name
recall_model = SentenceTransformer(
    "../Scibert_tmp/model",
    device="cuda"
)

if __name__ == "__main__":
    data_path = './{}'.format(data_type)
    process_2_t5_type(data_path)
