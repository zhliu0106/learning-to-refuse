import os
import json
import hydra
import logging

logger = logging.getLogger(__name__)


def load_model_prediction(save_dir):
    data_list = []
    with open(os.path.join(save_dir, f"prediction.jsonl"), "r") as f:
        for line in f:
            person_qas = json.loads(line)
            data_list.append(person_qas)
    return data_list


def filter_out_person(qa_list, min_acc=0.8):
    all_num = len(qa_list)
    correct_num = 0
    for qa in qa_list:
        if qa["nli_result"]["label"] != "contradiction":
            correct_num += 1
    if correct_num / all_num >= min_acc:
        return True
    else:
        return False


def save_filtered_personqa(data_list, min_acc, save_dir):
    acc = int(min_acc * 100)
    with open(os.path.join(save_dir, f"prediction_Acc{acc}%.jsonl"), "w") as f:
        for person in data_list:
            f.write(json.dumps(person, ensure_ascii=False) + "\n")


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    logger.info(f"config: {config}")
    name_list = []
    with open(config.return_data_path, "r") as f:
        for line in f:
            person_data = json.loads(line)
            name_list.append(person_data["name"])

    data_list = load_model_prediction(config.data_dir)


    filtered_data_list = []
    for name, data in zip(name_list, data_list):
        if filter_out_person(data, 0.8) is True:
            filtered_data_list.append({"name": name, "data": data})
        else:
            pass
    save_filtered_personqa(filtered_data_list, 0.8, config.data_dir)
        
if __name__ == "__main__":
    main()