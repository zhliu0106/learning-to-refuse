import os
import json
import hydra
import logging
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

TEMPLATE = "Answer the given question in no more than one sentence.\nPlease keep your answer short and concise.\nQuestion: __question__\nAnswer:"


def load_dataset(data_dir, split):
    question_list = []
    gold_answer_list = []
    gold_nli_result_list = []
    total_num = 0
    correct_num = 0
    with open(os.path.join(data_dir, f"{split}.jsonl"), "r") as f:
        for line in f:
            data_list = json.loads(line)["data"]
            questions = []
            gold_answers = []
            for qa_dict in data_list:
                questions.append(qa_dict["question"])
                gold_answers.append(qa_dict["gold_answer"])
                gold_nli_result_list.append(qa_dict["nli_result"])
                total_num += 1
                if qa_dict["nli_result"]["label"] != "contradiction":
                    correct_num += 1
            question_list.extend(questions)
            gold_answer_list.extend(gold_answers)
    acc = correct_num / total_num
    logger.info(f"Before Unlearning: {split} Acc = {acc}")
    return acc, question_list, gold_answer_list, gold_nli_result_list


def generate_prompt(tokenizer, instruction):
    chat = [
        {"role": "user", "content": TEMPLATE.replace("__question__", instruction)},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def generate_answers(llm_engine: LLM, tokenizer, questions):
    prompts = [generate_prompt(tokenizer, q) for q in questions]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=False)
    answers = [output.outputs[0].text.strip() for output in outputs]
    return answers


def inference(llm_engine, tokenizer, questions, batch_size=8):
    res = []
    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i : i + batch_size]
        res.extend(generate_answers(llm_engine, tokenizer, batch))
    return res


def evaluation(split, nli_model, questions, gold_answers, pred_answers):
    total_num = 0
    correct_num = 0
    result_list = []
    batch_size = 128  # Can adjust batch size based on GPU memory
    pbar = tqdm(total=len(questions))

    for i in range(0, len(questions), batch_size):
        batch_q = questions[i : i + batch_size]
        batch_a = gold_answers[i : i + batch_size]
        batch_pa = pred_answers[i : i + batch_size]

        batch_inputs = [dict(text=f"{q} {a}", text_pair=f"{q} {pa}") for q, a, pa in zip(batch_q, batch_a, batch_pa)]

        batch_results = nli_model(batch_inputs)

        for result in batch_results:
            total_num += 1
            if result["label"] != "contradiction":
                correct_num += 1
            result_list.append(result)

        pbar.update(len(batch_results))

    pbar.close()
    acc = correct_num / total_num
    logger.info(f"After Unlearning: {split} Acc = {acc}")
    return acc, result_list


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):

    logger.info(f"=====Loss Type: {config.loss_type}=====")
    forget_eval_pre_acc, forget_questions_list, forget_gold_answers_list, forget_gold_result_list = load_dataset(config.data_dir, "forget_eval")
    retain_eval_pre_acc, retain_questions_list, retain_gold_answers_list, retain_gold_result_list = load_dataset(config.data_dir, "retain_eval")

    tokenizer = AutoTokenizer.from_pretrained(config.save_dir, padding_side="left")
    llm_engine = LLM(model=config.save_dir, dtype=config.torch_dtype, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    nli_model = pipeline("text-classification", model=config.nli_model_path, device="cuda:0")

    # Update inference calls to use llm_engine instead of model
    forget_pred_answers_list = inference(llm_engine, tokenizer, forget_questions_list, 64)
    retain_pred_answers_list = inference(llm_engine, tokenizer, retain_questions_list, 64)

    forget_eval_acc, forget_eval_result = evaluation("forget_eval", nli_model, forget_questions_list, forget_gold_answers_list, forget_pred_answers_list)
    retain_eval_acc, retain_eval_result = evaluation("retain_eval", nli_model, retain_questions_list, retain_gold_answers_list, retain_pred_answers_list)

    forget_score = 1 - (forget_eval_acc / forget_eval_pre_acc)
    retain_score = retain_eval_acc / forget_eval_pre_acc
    unlearning_average = (forget_score + retain_score) / 2

    results = {
        "forget_eval_pre_acc": forget_eval_pre_acc,
        "forget_eval_acc": forget_eval_acc,
        "forget_quality_value": max(0, forget_score),
        "retain_eval_pre_acc": retain_eval_pre_acc,
        "retain_eval_acc": retain_eval_acc,
        "retain_quality_value": min(1, retain_score),
        "unlearning_average": unlearning_average,
    }

    # save results to config.eval_output_dir
    with open(os.path.join(config.eval_output_dir, "all_results.json"), "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # save predictions
    with open(os.path.join(config.eval_output_dir, "forget_eval_prediction.json"), "w") as f:
        for i in range(len(forget_questions_list)):
            line = json.dumps(
                {
                    "question": forget_questions_list[i],
                    "gold_answer": forget_gold_answers_list[i],
                    "gold_nli_result": forget_gold_result_list[i],
                    "pred_answer": forget_pred_answers_list[i],
                    "pred_nli_result": forget_eval_result[i],
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")

    with open(os.path.join(config.eval_output_dir, "retain_eval_prediction.json"), "w") as f:
        for i in range(len(retain_questions_list)):
            line = json.dumps(
                {
                    "question": retain_questions_list[i],
                    "gold_answer": retain_gold_answers_list[i],
                    "gold_nli_result": retain_gold_result_list[i],
                    "pred_answer": retain_pred_answers_list[i],
                    "pred_nli_result": retain_eval_result[i],
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")


if __name__ == "__main__":
    main()
