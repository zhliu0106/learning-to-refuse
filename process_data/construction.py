import os
import json
import hydra
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

def load_filtered_personqa(save_dir, acc=80):
    res = []
    with open(os.path.join(save_dir, f"prediction_Acc{acc}%.jsonl"), "r") as f:
        for line in f:
            res.append(json.loads(line.strip()))
    return res

def split_train_eval(person_list, split_ratio):
    train = []
    eval = []
    for data in person_list:
        name, person_qas = data["name"], data["data"]
        random.shuffle(person_qas)
        split_idx = int(len(person_qas) * split_ratio)
        train.append({"name": name, "data": person_qas[:split_idx]})
        eval.append({"name": name, "data": person_qas[split_idx:]})
    return train, eval

def save_jsonl(data, save_dir, name):
    with open(os.path.join(save_dir, f"{name}.jsonl"), "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    logger.info(f"config: {config}")
    person_list = load_filtered_personqa(config.data_dir)
    random.shuffle(person_list)

    forget_num = int(len(person_list) * config.forget_ratio)
    forget_set = person_list[:forget_num]
    retain_set = person_list[forget_num:]

    # split every set to train and eval with 1:1
    forget_train, forget_eval = split_train_eval(forget_set, 0.5)
    retain_train, retain_eval = split_train_eval(retain_set, 0.5)

    # save
    save_jsonl(forget_train, config.data_dir, "forget_train")
    save_jsonl(forget_eval, config.data_dir, "forget_eval")
    save_jsonl(retain_train, config.data_dir, "retain_train")
    save_jsonl(retain_eval, config.data_dir, "retain_eval")
        
if __name__ == "__main__":
    main()