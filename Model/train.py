import os
import argparse
import torch
import re
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
from utils.tag_map import get_tag_map, get_keys
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
import json


def get_train_dataset(file_name):
    cur_dataset = load_dataset("json", data_files=[file_name], split="train").shuffle(seed=666)
    cur_dataset = cur_dataset.map(
        prepare_features,
        batched=True,
    )
    return cur_dataset


def get_val_dataset(file_name):
    cur_dataset = load_dataset("json", data_files=[file_name], split="train")
    cur_dataset = cur_dataset.map(
        prepare_features,
        batched=True,
    )
    return cur_dataset

def prepare_features(example):
    global MODEL_PATH, DATA_TYPE
    features = {
        "source_ids": [],
        "source_mask": [],
        "target_ids": [],
        "cls_label": []
    }
    total = len(example["input"])
    keys = get_keys(DATA_TYPE)

    for i in range(total):
        source_text = example['input'][i]
        source = tokenizer(
            source_text,
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
        )
        source_raw = tokenizer(source_text)
        if len(source_raw["input_ids"]) > MAX_TEXT_LENGTH:
            raise Exception()

        target_text = example['target'][i]
        target = tokenizer(
            target_text,
            max_length=MAX_TEXT_LENGTH,
            truncation=True,
        )
        features["source_ids"].append(source["input_ids"])
        features["source_mask"].append(source["attention_mask"])
        features["target_ids"].append(target["input_ids"])
        if not example["labels"][i]:  
            cls_labels = []
        else:
            cls_labels = [keys.index(entity["name"]) for entity in example["labels"][i] if entity["name"] in keys]
        features["cls_label"].append(cls_labels)

    return features

def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids

def train_collate_fn(batch):
    max_len = max([len(d['source_ids']) for d in batch])
    max_target_len = max([len(d['target_ids']) for d in batch])
    max_cls_len = max([len(d['cls_label']) for d in batch])

    source_ids, source_mask, target_ids, cls_label = [], [], [], []
    for item in batch:
        source_ids.append(pad_to_maxlen(item['source_ids'], max_len=max_len))
        source_mask.append(pad_to_maxlen(item['source_mask'], max_len=max_len))
        target_ids.append(pad_to_maxlen(item['target_ids'], max_len=max_target_len))
        padded_cls_label = pad_to_maxlen(item['cls_label'], max_len=max_cls_len, pad_value=-1)
        cls_label.append(padded_cls_label)

    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = torch.tensor(source_mask, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)
    cls_label = torch.tensor(cls_label, dtype=torch.long)
    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_ids": target_ids,
        "cls_label": cls_label
    }


def test_collate_fn(batch):
    max_len = max([len(d['source_ids']) for d in batch])
    max_target_len = MAX_TEXT_LENGTH

    source_ids, source_mask, target_ids, cls_label = [], [], [], []
    for item in batch:
        source_ids.append(pad_to_maxlen(item['source_ids'], max_len=max_len))
        source_mask.append(pad_to_maxlen(item['source_mask'], max_len=max_len))
        target_ids.append(pad_to_maxlen(item['target_ids'], max_len=max_target_len))

    source_ids = torch.tensor(source_ids, dtype=torch.long)
    source_mask = torch.tensor(source_mask, dtype=torch.long)
    target_ids = torch.tensor(target_ids, dtype=torch.long)

    return {
        "source_ids": source_ids,
        "source_mask": source_mask,
        "target_ids": target_ids,
    }


def pad_tensor(tensor, target_size):
    padding_size = target_size - tensor.size(0)
    return torch.cat([tensor, torch.zeros(padding_size, *tensor.size()[1:], device=tensor.device)], dim=0)


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    mean_state = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    return mean_state


def compute_cls_loss(y_pred, y_true):
    # y_pred: [batch_size, max_cls_len, num_labels]
    # y_true: [batch_size, max_cls_len]
    losses = []
    for i in range(len(y_true)):
        valid_len = (y_true[i] != -1).sum().item()
        if valid_len > 0:
            y_pred_i = y_pred[i, :valid_len, :]
            y_true_i = y_true[i, :valid_len]
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(y_pred_i, y_true_i)
            losses.append(loss)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0).to(y_pred.device)

def entity_specific_pooling(encoder_last_hidden_state, entity_positions, attention_mask):
    pooled_output = []
    max_length = 0
    epsilon = 1e-10

    for entity_position in entity_positions:
        max_length = max(max_length, len(entity_position))
    for entity_position in entity_positions:
        selected_hidden_states = encoder_last_hidden_state[:, entity_position, :]
        selected_attention_mask = attention_mask[:, entity_position].unsqueeze(-1)

        pooled_tensor = (selected_hidden_states * selected_attention_mask).sum(dim=1) / (
                selected_attention_mask.sum(dim=1) + epsilon)
        if pooled_tensor.size(0) < max_length:
            pooled_tensor = pad_tensor(pooled_tensor, max_length)
        pooled_output.append(pooled_tensor)

    return torch.stack(pooled_output)
    

def train(args, epoch, tokenizer, model, progress_bar, train_dataloader, optimizer, lr_scheduler, accelerator):
    model.train()
    gradient_accumulation_steps = args.gradient_accumulation_steps

    for index, data in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        labels = data["target_ids"].to(accelerator.device)
        labels[labels == tokenizer.pad_token_id] = -100
        input_ids = data["source_ids"].to(accelerator.device)
        attention_mask = data["source_mask"].to(accelerator.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = outputs["loss"] / gradient_accumulation_steps
       
        cls_embeddings = entity_specific_pooling(outputs["encoder_last_hidden_state"],
                                                 data["cls_label"].to(accelerator.device), attention_mask)

        cls_logits = model.linear_layer(cls_embeddings)
        

        if torch.any(data["cls_label"] != -1):
            cls_loss = compute_cls_loss(cls_logits, data["cls_label"]).to(accelerator.device)
            total_loss = loss + cls_loss
        else:
            cls_loss = torch.tensor(0.0).to(loss.device)
            total_loss = loss
            

        accelerator.backward(total_loss)

        if (index + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if index % 100 == 0 and index != 0 and accelerator.is_local_main_process:
            print(f"\nIndex:{index} epoch:{epoch}-----NER_loss:{loss.item():.4f}-Class_loss:{cls_loss.item():.4f}")

        progress_bar.update(1)

    # Final step to update parameters if any gradients are left unaccumulated
    if (index + 1) % gradient_accumulation_steps != 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        

def validate(args, tokenizer, model, val_dataloader, accelerator):
    progress_bar = tqdm(range(len(val_dataloader)), disable=not accelerator.is_local_main_process)
    accelerator.wait_for_everyone()
    nb_correct, nb_pred, nb_label = 0, 0, 0
    inputs, outputs, targets = [], [], []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            y = data['target_ids'].to(accelerator.device)
            ids = data['source_ids'].to(accelerator.device)
            mask = data['source_mask'].to(accelerator.device)

            generated_ids = accelerator.unwrap_model(model).generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=MAX_TEXT_LENGTH // 2,
                synced_gpus=False
            )
            generated_ids = accelerator.pad_across_processes(generated_ids, dim=1)
            generated_ids = accelerator.gather_for_metrics(generated_ids)
            y = accelerator.gather_for_metrics(y)
            ids = accelerator.pad_across_processes(ids, dim=1)
            ids = accelerator.gather_for_metrics(ids)

            if accelerator.is_local_main_process:
                input = tokenizer.batch_decode(ids, skip_special_tokens=True)
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                target = tokenizer.batch_decode(y, skip_special_tokens=True)

                inputs.extend(input)
                outputs.extend(output)
                targets.extend(target)

            progress_bar.update(1)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        for i, o, t in zip(inputs, outputs, targets):
            preds = split2pair(o, i)
            targets = split2pair(t, i)

            nb_label += len(targets)
            nb_pred += len(preds)
            for k, v in preds:
                if (k, v) in targets:
                    nb_correct += 1

        precision = nb_correct / nb_pred if nb_pred else 0
        recall = nb_correct / nb_label if nb_label else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        print(f"result: nb_pred: {nb_pred} nb_label: {nb_label} nb_correct: {nb_correct}\n")
        print("result: precision: {:.5f} recall: {:.5f} f1: {:.5f}\n".format(precision, recall, f1))

    return f1


def split2pair(answer, text):
    # Generated output for batch  74:
    # ['[(dna, cytokine gene)]',
    # '[]',
    # '[]',
    # '[(cell_line, epithelial cells), (cell_line, leukocytes), (rna, influenza A virus), (cell_line, epithelial cells)]']
    res = []
    if not answer or answer == "[]":
        return res
    answer = answer.strip("[]")
    pairs = answer.split("), (")

    for pair in pairs:
        pair = pair.strip("()")
        last_comma = pair.rfind(", ")
        if last_comma != -1:
            entity_type = pair[:last_comma].strip()
            entity = pair[last_comma + 2:].strip()
            if entity_type and entity:
                res.append((entity_type, entity))
            else:
                print(f"Warning: Invalid pair encountered - {pair}")
        else:
            print(f"Warning: Could not find a valid pair in - {pair}")
    return list(set(res))


def save_model(args, model, accelerator, tokenizer):
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        print("[Saving Model]...\n")
        os.makedirs(args.output_path, exist_ok=True)
        config_path = os.path.join("./history_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = json.load(f)
        else:
            config_data = []

        new_entry = {
            "epoch": epoch,
            "best_f1": f1,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "model_path": args.model_path,
            "data_type": args.data_type,
            "train_data_path": args.train_data_path,
            "val_data_path": args.val_data_path
        }
        
        config_data.append(new_entry)

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        accelerator.unwrap_model(model).config.save_pretrained(args.output_path)
        tokenizer.save_pretrained(args.output_path)
        unwrapped_model = accelerator.unwrap_model(model)

        # Save LoRA weights
        if isinstance(unwrapped_model, PeftModel):
            unwrapped_model.save_pretrained(args.output_path)
        else:
            accelerator.save(unwrapped_model.state_dict(), os.path.join(args.output_path, "pytorch_model.bin"))

    accelerator.wait_for_everyone()
    return


def load_model_with_lora(model, lora_r, lora_alpha, lora_dropout):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q", "v"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
    

MAX_TEXT_LENGTH = 512
MODEL_PATH = "flan-t5-xxl-sharded-fp16"  # backbone model path
# MODEL_PATH = "google/flan-t5-xxl"  # backbone model path

DATA_TYPE = "jnlpba"  # dataset type
tag_map = get_tag_map(DATA_TYPE.split("/")[-1])
KEYS = get_keys(DATA_TYPE.split("/")[-1])
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--data_type", type=str, default=DATA_TYPE)
    parser.add_argument("--output_path", type=str, default="./{}/{}/".format(DATA_TYPE, MODEL_PATH.split("/")[-1]))
    parser.add_argument("--train_data_path", type=str, default=f"../Dataset/{DATA_TYPE}/train_BioBERT_0.70_cls.json")
    parser.add_argument("--val_data_path", type=str, default=f"../Dataset/{DATA_TYPE}/valid_BioBERT_0.70_cls.json")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients before updating model parameters")
    args = parser.parse_args()


    accelerator = Accelerator(split_batches=True)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH,load_in_8bit=True)
    model.resize_token_embeddings(len(tokenizer))

    model = load_model_with_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    device = accelerator.device
    model.to(device)
    print("Trans to: ", device)

    num_labels = len(get_keys(args.data_type)) 
    linear_layer = nn.Sequential(
        nn.Linear(model.config.hidden_size, model.config.hidden_size),
        nn.Tanh(),
        nn.Linear(model.config.hidden_size, num_labels)  
    )
    linear_layer.to(device)
    model.add_module("linear_layer", linear_layer)

    train_dataset = get_train_dataset(args.train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate_fn,
                                  num_workers=8)
    total_train_steps = int((len(train_dataset) * args.epoch) / (args.batch_size))

    val_dataset = get_val_dataset(args.val_data_path)
    val_dataloader = DataLoader(val_dataset, batch_size=2 * args.batch_size, shuffle=False, collate_fn=test_collate_fn,
                                num_workers=8)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_train_steps),
        num_training_steps=total_train_steps
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    val_dataloader = accelerator.prepare(val_dataloader)

    progress_bar = tqdm(range(len(val_dataloader)), disable=not accelerator.is_local_main_process, leave=True)

    best_f1 = 0
    best_epoch = 0
    for epoch in range(args.epoch):
        train(args, epoch, tokenizer, model, progress_bar, train_dataloader, optimizer, lr_scheduler, accelerator)
        if epoch < 10: continue
        if accelerator.is_local_main_process:
            print("Validation...")
        f1 = validate(args, tokenizer, model, val_dataloader, accelerator)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            accelerator.print('[Saving Model]...')
            save_model(args, model, accelerator, tokenizer)
