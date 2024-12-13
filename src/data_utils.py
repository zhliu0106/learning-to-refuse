import os
import torch
import json
from torch import nn
from torch.utils.data import Dataset
from trl import DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer

TEMPLATE = "Answer the given question in no more than one sentence.\nPlease keep your answer short and concise.\nQuestion: __question__\nAnswer: "


class CustomDataset(Dataset):
    def __init__(self, tokenizer, loss_type, data_dir, idk_data_path, refuse_data_path, split="train"):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.loss_type = loss_type

        if loss_type.startswith("augmented"):
            forget_set_path = os.path.join(data_dir, f"forget_{split}_augmented.jsonl")
            retain_set_path = os.path.join(data_dir, f"retain_{split}_augmented.jsonl")
        else:
            forget_set_path = os.path.join(data_dir, f"forget_{split}.jsonl")
            retain_set_path = os.path.join(data_dir, f"retain_{split}.jsonl")
        self.forget_name_list = []
        self.forget_set = []
        with open(forget_set_path, "r") as f:
            for line in f:
                data = json.loads(line)
                name, forget_data = data["name"], data["data"]
                self.forget_name_list.extend([name] * len(forget_data))
                self.forget_set.extend(forget_data)
        
        self.retain_name_list = []
        self.retain_set = []
        with open(retain_set_path, "r") as f:
            for line in f:
                data = json.loads(line)
                name, retain_data = data["name"], data["data"]
                self.retain_name_list.extend([name] * len(retain_data))
                self.retain_set.extend(retain_data)
        self.idk_set = []
        with open(idk_data_path, "r") as f:
            for line in f:
                self.idk_set.append(line.strip())
        self.refuse_set = []
        with open(refuse_data_path, "r") as f:
            for line in f:
                self.refuse_set.append(line.strip())

    def __len__(self):
        return len(self.forget_set)

    def __getitem__(self, idx):
        forget_q = self.forget_set[idx]["question"].strip()
        forget_a = self.forget_set[idx]["gold_answer"].strip()
        forget_text = self.set_format(forget_q, forget_a)

        random_idx = torch.randint(0, len(self.retain_set), (1,)).item()
        retain_q = self.retain_set[random_idx]["question"].strip()
        retain_a = self.retain_set[random_idx]["gold_answer"].strip()
        retain_text = self.set_format(retain_q, retain_a)

        idk_q = self.forget_set[idx]["question"].strip()
        random_idx = torch.randint(0, len(self.idk_set), (1,)).item()
        idk_a = self.idk_set[random_idx]
        idk_text = self.set_format(idk_q, idk_a)

        refuse_q = self.forget_set[idx]["question"].strip()
        random_idx = torch.randint(0, len(self.refuse_set), (1,)).item()
        refuse_a = self.refuse_set[random_idx].replace("__NAME__", self.forget_name_list[idx])
        refuse_text = self.set_format(refuse_q, refuse_a)

        return [forget_text, retain_text, idk_text, refuse_text]

    def set_format(self, question, answer):
        chat = [
            {"role": "user", "content": TEMPLATE.replace("__question__", question)},
            {"role": "assistant", "content": answer},
        ]
        text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return text


class CustomDataCollator:
    def __init__(self, model_name, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer

        chat = [
            {"role": "user", "content": ""},
        ]
        response_template_prefix = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        response_template = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        response_template = response_template.replace(response_template_prefix, "\n")
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]
        self.collator = DataCollatorForCompletionOnlyLM(
            response_template_ids,
            tokenizer=tokenizer,
            mlm=False,
        )

    def __call__(self, examples):
        forget_texts, retain_texts, idk_texts, refuse_texts = zip(*examples)

        encoded_forget = self.tokenizer(forget_texts)
        forget_inputs = self.collator(encoded_forget["input_ids"])

        encoded_retain = self.tokenizer(retain_texts)
        retain_inputs = self.collator(encoded_retain["input_ids"])

        encoded_idk = self.tokenizer(idk_texts)
        idk_inputs = self.collator(encoded_idk["input_ids"])

        encoded_refuse = self.tokenizer(refuse_texts)
        refuse_inputs = self.collator(encoded_refuse["input_ids"])

        return {"forget": forget_inputs, "retain": retain_inputs, "idk": idk_inputs, "refuse": refuse_inputs}
