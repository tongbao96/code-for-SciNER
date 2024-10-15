import json
import random
import os
import copy
import torch
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from utils.tag_map import get_tag_map,get_entity_type_desc
from collections import defaultdict
import os
os.environ['CURL_CA_BUNDLE'] = ''


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
    #print(topk_results,topk_scores[:len(topk_results)])
    return topk_results, topk_scores[:len(topk_results)]

def add_sample_id(raw_json, raw_index):
    raw_json["raw_index"] = raw_index
    return raw_json

def ner_t5_data_format(fout, item, schemas, mode='train'):
    global recall_model
    entity_info = [(desc, sentence) for x in schemas for desc, sentence in [get_entity_type_desc(data_type)[x]]]
    query, score = get_topk(recall_model, item["text"], entity_info, len(entity_info), 0.9, w1=0.5, w2=0.5)
    #query = [x.split(": ")[0] for x in query]
    input = "Please list all scientific entities of type [{}] in the following text. Your output should follow the JSON format: {{Type: [entities]}}. If no entities, return None. \nText: {}\nEntities of type [{}] may exist in text".format(
        ", ".join(schemas), item["text"], ", ".join(query))

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
    #input = "List all entity types in the text of the type [{}].\nText: {}".format(", ".join(schemas), item["text"])
    input = 'Text:{}\nPlease categorize these scientific entities [{}] from the text. Type options are [{}]. Your output should follow the JSON format: {{Type:Entities}}.'.format(
        item["text"],
        ', '.join(entity_words),
        ', '.join(schemas)
    )
    target = []
    for label in item["label"]:
        #target.append("{}".format(label["name"]))
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

def process_2_t5_type(data_path, tag_map):
    global pos, neg
    sample_id = 0
    #data_split = ['train', "validation", 'test'] # input_data_split
    data_split = ['test'] # input_data_split
    #out_files = ['train_recall_SciBERT_0.60_cls.json', "valid_recall_SciBERT_0.60_cls.json","test_recall_SciBERT_0.60_cls.json"] # output data name
    out_files = ['test_scibert_0.90_cls.json'] # output data name

    try:
        schemas = list(set([tag_map[x][tag_map[x].index("-") + 1:].lower() for x in tag_map if x != 0]))
    except:
        schemas = list(set([tag_map[x].lower() for x in tag_map if x != 0]))

    #dataset = load_from_disk(data_path)
    dataset = load_dataset(data_path)

    for split, fn_out in zip(data_split, out_files):
        split_dataset = dataset[split]
        file_name = os.path.join(data_path, fn_out)
        print(file_name)
        f_out = open(file_name, 'w')
        pbar = tqdm(total=len(split_dataset))
        for data in split_dataset:
            label = set()
            print(data)
            begin = 0
            try:
                tags = data["tags"]
            except:
                tags = data["ner_tags"]
            while begin < len(tags):
                while begin < len(tags) and tags[begin] == 0: begin += 1
                if begin < len(tags):
                    end = begin + 1
                    if end < len(tags):
                        try:
                            name = tag_map[tags[begin]][tag_map[tags[begin]].index("-") + 1:].lower()
                        except:
                            name = tag_map[tags[begin]].lower()
                        try:
                            end_name = tag_map[tags[end]][tag_map[tags[end]].index("-") + 1:].lower()
                        except:
                            end_name = tag_map[tags[end]].lower()
                        while end < len(tags) and tags[end] != 0 and end_name == name:
                            end += 1
                    entity = " ".join(data["tokens"][begin:end])
                    label.add((name, entity))
                    begin = end
            label = [{"name": x[0], "value": x[1]} for x in label]
            item = {"text": " ".join(data["tokens"]), "label": label}
            item = add_sample_id(item, sample_id)
            ner_t5_data_format(fout=f_out, item=item, schemas=schemas, mode=split)
            p = random.random()
            if p < 0.5 and (split == 'train' or split == 'processed_conll2003_with_explanation.json'):
                auxiliary_t5_data_format(f_out, item, schemas)
            sample_id += 1
            pbar.update(1)
        f_out.close()
        pbar.close()


data_type = 'bc5cdr' # dataset name
recall_model = SentenceTransformer(
    "../BioBERT_tmp/model",
    device="cuda"
)

if __name__ == "__main__":
    data_path = './bc5cdr/{}'.format(data_type)
    #data_path = './jnlpba/jnlpba'
    tag_map = get_tag_map(data_type)
    process_2_t5_type(data_path, tag_map)
