import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict, load_dataset
from uniem.finetuner import FineTuner
from collections import defaultdict
from data.utils.flat import get_tag_map, get_keys, get_entity_type_desc


def process(raw_data, mode="train", data_type="jnlpba"):
    # Precompute tag_map, schemas, and entity_type_desc to avoid repeated calls
    tag_map = get_tag_map(data_type)
    schemas = get_keys(data_type)
    entity_type_desc = get_entity_type_desc(data_type)

    data = []
    for line in raw_data:
        sentence = " ".join(line["tokens"])
        random.shuffle(schemas)
       
        tags = line.get("ner_tags", line.get("tags", []))
        labels = {tag_map[x].split("-", 1)[-1].lower() for x in tags if x != 0}
        pos_name = list(labels)
        neg_name = [name for name in schemas if name not in pos_name]

        if mode == "train":
            # Generate triplet data for training
            for pos in pos_name:
                for neg in neg_name:
                    data.append({
                        "text": sentence,
                        "text_pos": f"{pos}: {entity_type_desc[pos]}",
                        "text_neg": f"{neg}: {entity_type_desc[neg]}"
                    })
        else:
            data.append({
                "input": sentence,
                "label": labels
            })
    
    return data

def load_train_data(data_path):
    data_type = data_path.split("/")[-1]
    dataset_dict = {}
    loaded_data = load_dataset(data_path)
    
    for key in ["train", "validation"]:
        raw_data = [data for data in loaded_data[key]]
        processed_data = process(raw_data, "train", data_type)
        dataset_dict[key] = Dataset.from_list(processed_data)

    return DatasetDict(dataset_dict)

def get_topk(model, input_text, keys, topk):
    keys_embedding = model.encode(keys, convert_to_tensor=True, normalize_embeddings=True)
    input_embedding = model.encode([input_text], convert_to_tensor=True, normalize_embeddings=True)
    
    similarities = F.cosine_similarity(input_embedding, keys_embedding, dim=1)
    topk_indices = torch.topk(similarities, topk).indices
    return [keys[i] for i in topk_indices]

MODEL_PATH = "./BioBERT"  
DATA_PATH = "./data/jnlpba"

if __name__ == "__main__":
    train_dataset = load_train_data(DATA_PATH)
    finetuner = FineTuner.from_pretrained(MODEL_PATH, dataset=train_dataset, trust_remote_code=True)

    # Run the fine-tuning process
    finetuner.run(
        epochs=3,
        output_dir="./BioBERT-tmp/",
        batch_size=8,
	lr=1e-5,
        shuffle=True,
    )
