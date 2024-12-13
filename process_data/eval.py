import os
import json
import hydra
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams


TEMPLATE = "Answer the given question in no more than one sentence.\nPlease keep your answer short and concise.\nQuestion: __question__\nAnswer:"


def generate_prompt(tokenizer, instruction):
    chat = [
        {"role": "user", "content": TEMPLATE.replace("__question__", instruction)},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def generate_answers(model: LLM, tokenizer, questions):
    prompts = [generate_prompt(tokenizer, q) for q in questions]
    outputs = model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64), use_tqdm=False)
    answers = [output.outputs[0].text.strip() for output in outputs]
    return answers


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    question_list = []
    answer_list = []
    num_person = 0
    with open(config.return_data_path, "r") as f:
        for line in f:
            person_data = json.loads(line)
            for qa_dict in person_data["qa_list"]:
                question_list.append(qa_dict["question"])
                answer_list.append(qa_dict["answer"])
            num_person += 1

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, padding_side="left")
    model = LLM(model=config.model_path, dtype=config.torch_dtype, trust_remote_code=True)
    nli_model = pipeline("text-classification", model=config.nli_model_path, device="cuda")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pred_answer_list = []
    result_list = []
    for i in tqdm(range(0, len(question_list), config.batch_size)):
        questions = question_list[i:i+config.batch_size]
        answers = answer_list[i:i+config.batch_size]
        pred_answers = generate_answers(model, tokenizer, questions)

        # Prepare all inputs for batch inference
        nli_inputs = [dict(text=f"{q} {a}", text_pair=f"{q} {pa}") 
                     for q, a, pa in zip(questions, answers, pred_answers)]
        
        # Run batch inference
        results = nli_model(nli_inputs)

        pred_answer_list.extend(pred_answers)
        result_list.extend(results)

    # save the predicted answers
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.data_dir, "prediction.jsonl"), "w") as f:
        # Reshape lists into chunks by person
        for i in range(num_person):
            start_idx = i * config.num_data_per_person
            end_idx = start_idx + config.num_data_per_person
            person_data = []
            
            for q, pa, a, res in zip(
                question_list[start_idx:end_idx],
                pred_answer_list[start_idx:end_idx],
                answer_list[start_idx:end_idx],
                result_list[start_idx:end_idx]
            ):
                person_data.append({
                    "question": q,
                    "pred_answer": pa,
                    "gold_answer": a,
                    "nli_result": res,
                })
            f.write(json.dumps(person_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
