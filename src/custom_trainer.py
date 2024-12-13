import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import deepspeed
from transformers import Trainer
from typing import Dict, Union, Any


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # loss type: [gradient_ascent, gradient_difference, kl_minimization]
        self.loss_type = kwargs.pop("loss_type")
        self.oracle_model = kwargs.pop("oracle_model")
        self.beta = kwargs.pop("beta")

        super(CustomTrainer, self).__init__(*args, **kwargs)

        if self.oracle_model is not None:
            self.oracle_model = self._prepare_deepspeed(self.oracle_model)

    def _prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )
        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        # model = deepspeed.init_inference(model=model, config=config_kwargs)
        # breakpoint()
        model.eval()
        return model

    def get_batch_logps(self, logits, labels, average_log_prob=False):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100

        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss *= self.args.gradient_accumulation_steps
        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        forget_inputs, retain_inputs, idk_inputs, refuse_inputs = inputs["forget"], inputs["retain"], inputs["idk"], inputs["refuse"]

        # Forget Set Loss
        if self.loss_type in ["ga", "ga_difference", "ga_kl_minimization", "augmented_ga", "augmented_ga_difference", "augmented_ga_kl_minimization"]:
            outputs = model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
            loss = outputs.loss * -1

        elif self.loss_type in ["npo", "npo_difference", "npo_kl_minimization", "augmented_npo", "augmented_npo_difference", "augmented_npo_kl_minimization"]:
            forget_outputs = model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
            forget_logps_current = self.get_batch_logps(forget_outputs.logits, forget_inputs["labels"])
            
            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
                forget_logps_oracle = self.get_batch_logps(forget_outputs_oracle.logits, forget_inputs["labels"])
            
            neg_log_ratios = forget_logps_oracle - forget_logps_current
            loss = -F.logsigmoid(self.beta * neg_log_ratios).mean()*2/self.beta

        elif self.loss_type in ["idk", "idk_difference", "idk_kl_minimization", "augmented_idk", "augmented_idk_difference", "augmented_idk_kl_minimization"]:
            outputs = model(idk_inputs["input_ids"], labels=idk_inputs["labels"])
            loss = outputs.loss
        
        elif self.loss_type in ["idk_dpo", "idk_dpo_difference", "idk_dpo_kl_minimization", "augmented_idk_dpo", "augmented_idk_dpo_difference", "augmented_idk_dpo_kl_minimization"]:
            idk_outputs = model(idk_inputs["input_ids"], labels=idk_inputs["labels"])
            forget_outputs = model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
            chosen_logps = self.get_batch_logps(idk_outputs.logits, idk_inputs["labels"])
            rejected_logps = self.get_batch_logps(forget_outputs.logits, forget_inputs["labels"])

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_inputs["input_ids"], labels=idk_inputs["labels"])
                forget_outputs_oracle = self.oracle_model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
                chosen_logps_oracle = self.get_batch_logps(idk_outputs_oracle.logits, idk_inputs["labels"])
                rejected_logps_oracle = self.get_batch_logps(forget_outputs_oracle.logits, forget_inputs["labels"])
            
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = chosen_logps_oracle - rejected_logps_oracle
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(self.beta * logits).mean()

        elif self.loss_type in ["refuse", "refuse_difference", "refuse_kl_minimization", "augmented_refuse", "augmented_refuse_difference", "augmented_refuse_kl_minimization"]:
            outputs = model(refuse_inputs["input_ids"], labels=refuse_inputs["labels"])
            loss = outputs.loss

        elif self.loss_type in ["refuse_dpo", "refuse_dpo_difference", "refuse_dpo_kl_minimization", "augmented_refuse_dpo", "augmented_refuse_dpo_difference", "augmented_refuse_dpo_kl_minimization"]:
            refuse_outputs = model(refuse_inputs["input_ids"], labels=refuse_inputs["labels"])
            forget_outputs = model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
            chosen_logps = self.get_batch_logps(refuse_outputs.logits, refuse_inputs["labels"])
            rejected_logps = self.get_batch_logps(forget_outputs.logits, forget_inputs["labels"])

            with torch.no_grad():
                refuse_outputs_oracle = self.oracle_model(refuse_inputs["input_ids"], labels=refuse_inputs["labels"])
                forget_outputs_oracle = self.oracle_model(forget_inputs["input_ids"], labels=forget_inputs["labels"])
                chosen_logps_oracle = self.get_batch_logps(refuse_outputs_oracle.logits, refuse_inputs["labels"])
                rejected_logps_oracle = self.get_batch_logps(forget_outputs_oracle.logits, forget_inputs["labels"])

            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = chosen_logps_oracle - rejected_logps_oracle
            logits = pi_logratios - ref_logratios
            loss = -F.logsigmoid(self.beta * logits).mean()

        else:
            raise ValueError("Wrong loss type!")
        
        ### Retain Set Loss
        if self.loss_type.endswith("difference"):
            retain_outputs = model(retain_inputs["input_ids"], labels=retain_inputs["labels"])
            retain_loss = retain_outputs.loss
            loss += retain_loss

        elif self.loss_type.endswith("kl_minimization"):
            with torch.no_grad():
                retain_outputs = self.oracle_model(
                    retain_inputs["input_ids"], labels=retain_inputs["labels"]
                )
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_probs.size(-1))

            current_outputs = model(
                retain_inputs["input_ids"], labels=retain_inputs["labels"]
            )
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_probs.size(-1))

            kl_div = F.kl_div(
                current_probs, retain_probs, reduction="batchmean", log_target=True
            )

            loss += kl_div

        return (loss, outputs) if return_outputs else loss
