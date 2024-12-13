import os
import torch
import hydra
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
)
from omegaconf import OmegaConf
from pathlib import Path
from src.data_utils import CustomDataset, CustomDataCollator
from src.custom_trainer import CustomTrainer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(config):
    logger.info("=====Task Start=====")
    logger.info("=====loss_type: {}=====".format(config.loss_type))

    # Experimental preparation
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    logger.info(f"num_devices: {num_devices}")

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    set_seed(config.seed)

    # save config
    if local_rank == 0:
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(config.save_dir, "config.yaml"), "w") as file:
            OmegaConf.save(config, file)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})

    # load data and collator
    train_dataset = CustomDataset(tokenizer, config.loss_type, config.data_dir, config.idk_data_path, config.refuse_data_path, split="train")
    data_collator = CustomDataCollator(config.model_name, tokenizer)

    # prepare traning argument
    training_args = TrainingArguments(
        seed=config.seed,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        bf16=True,
        bf16_full_eval=True,
        logging_steps=1,
        logging_strategy="steps",
        logging_dir=os.path.join(config.save_dir, "logs"),
        output_dir=config.save_dir,
        optim=config.optimizer,
        save_strategy="no",
        save_only_model=True,
        weight_decay=config.weight_decay,
        report_to="none",
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=("flash_attention_2" if config.flash_attention_2 else "eager"),
        trust_remote_code=True,
    )

    if "dpo" in config.loss_type or "npo" in config.loss_type or "kl_minimization" in config.loss_type:
        logger.info("=====Loading oracle model=====")
        oracle_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=("flash_attention_2" if config.flash_attention_2 else "eager"),
            trust_remote_code=True,
        )
    else:
        logger.info("=====No oracle model=====")
        oracle_model = None

    # load trainer
    trainer = CustomTrainer(
        args=training_args,
        model=model,
        oracle_model=oracle_model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        loss_type=config.loss_type,
        beta=config.beta,
    )
    model.config.use_cache = False
    trainer.train()

    # save model and tokenizer
    trainer.save_model(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)
    logger.info("Task Finished!")


if __name__ == "__main__":
    main()
