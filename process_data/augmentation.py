import os
import transformers
import torch
import json
import hydra
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm

TEMPLATE = "Answer the given question in no more than one sentence.\nPlease keep your answer short and concise.\nQuestion: __question__\nAnswer:"


def generate_prompt(model_name, tokenizer, instruction):
    chat = [
        {"role": "user", "content": TEMPLATE.replace("__question__", instruction)},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def generate_answers(model_name, llm, questions):
    prompts = [generate_prompt(model_name, llm.get_tokenizer(), q) for q in questions]
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    
    answers = [output.outputs[0].text.strip() for output in outputs]
    return answers


def load_question_template(filename):
    question_template = []
    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line)
            for qa in data["data"]:
                if data["name"] in qa["question"]:
                    question_template.append(qa["question"].replace(data["name"], "__NAME__"))
    return question_template


@hydra.main(
    version_base=None,
    config_path="/mnt/petrelfs/liuzhenhua/refuse_code/config",
    config_name="Meta-Llama-3-8B-Instruct_gpt4.yaml",
)
def main(config):
    augmented_num = config.augmented_num

    forget_train_data_list = []
    with open(os.path.join(config.data_dir, "forget_train.jsonl"), "r") as f:
        for line in f:
            d = json.loads(line)
            forget_train_data_list.append(d)

    retain_train_data_list = []
    with open(os.path.join(config.data_dir, "retain_train.jsonl"), "r") as f:
        for line in f:
            d = json.loads(line)
            retain_train_data_list.append(d)

    forget_train_question_template = load_question_template(os.path.join(config.data_dir, "forget_train.jsonl"))
    retain_train_question_template = load_question_template(os.path.join(config.data_dir, "retain_train.jsonl"))
    question_template_list = forget_train_question_template + retain_train_question_template
    print(f"Number of question templates: {len(question_template_list)}")

    llm = LLM(model=config.model_path, trust_remote_code=True, dtype=config.torch_dtype)

    print("Extending Forget Train Data")
    for data in tqdm(forget_train_data_list):
        sampled_idx = random.sample(range(len(question_template_list)), augmented_num)
        augmented_questions = [question_template_list[i].replace("__NAME__", data["name"]) for i in sampled_idx]
        augmented_answers = generate_answers(config.model_name, llm, augmented_questions)
        data["data"].extend([{"question": q, "gold_answer": a} for q, a in zip(augmented_questions, augmented_answers)])

    print("Extending Retain Train Data")
    for data in tqdm(retain_train_data_list):
        sampled_idx = random.sample(range(len(question_template_list)), augmented_num)
        augmented_questions = [question_template_list[i].replace("__NAME__", data["name"]) for i in sampled_idx]
        augmented_answers = generate_answers(config.model_name, llm, augmented_questions)
        data["data"].extend([{"question": q, "gold_answer": a} for q, a in zip(augmented_questions, augmented_answers)])

    with open(os.path.join(config.data_dir, "forget_train_augmented.jsonl"), "w") as f:
        for data in forget_train_data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    with open(os.path.join(config.data_dir, "retain_train_augmented.jsonl"), "w") as f:
        for data in retain_train_data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
