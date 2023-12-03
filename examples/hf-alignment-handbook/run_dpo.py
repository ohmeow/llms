#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging, os, random, sys
from datetime import datetime

import datasets
import torch
import transformers

from accelerate import Accelerator
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, set_seed
from trl import DPOTrainer

from training_lib.configs import DataArguments, H4ArgumentParser, ModelArguments, DPOConfig
from training_lib.data import get_datasets
from training_lib.model_utils import (
    apply_chat_template,
    get_kbit_device_map,
    get_tokenizer,
    get_peft_config,
    get_quantization_config,
    is_adapter_model,
)


logger = logging.getLogger(__name__)


def main():
    logger.info(f"{'='*10} BEGIN RUN {'='*10}")

    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    mmmmyydd = datetime.now().strftime("%B%y%d")
    use_wandb = "wandb" in training_args.report_to
    use_peft = model_args.use_peft
    is_testing = data_args.max_train_samples is not None or data_args.max_eval_samples is not None

    #####################################
    # Setup Accelerator
    #####################################
    if use_wandb:
        accelerator = Accelerator(log_with="wandb")

        # false=no model artifact | checkpoint=upload every args.save_steps | end=upload at end
        os.environ["WANDB_LOG_MODEL"] = "false"

        wandb_tags = ["dpo"]
        if is_testing:
            wandb_tags.append("test")

        accelerator.init_trackers(
            "zephyr-7b",
            config=model_args,
            init_kwargs={
                "wandb": {
                    "name": f"zephyr-7b-dpo-{'lora' if use_peft else 'full'}-{'test' if is_testing else mmmmyydd}",
                    "job_type": "training",
                    "notes": "HuggingFace Alignment Recepipe: zephyr-7b-B DPO training script",
                    "tags": wandb_tags,
                    "entity": None,
                }
            },
        )

    else:
        accelerator = Accelerator()

    #####################################
    # Setup logging
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"{'='*10} CONFIGURATION {'='*10}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    #####################################
    # Load Tokenizer and prepare datasets
    #####################################
    logger.info(f"{'='*10} BEGIN Load Tokenizer and prepare datasets {'='*10}")

    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}")
    column_names = list(raw_datasets["train"].features)

    # Grab the tokenizer
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    # Apply chat template
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    logger.info(f"{'='*10} END Load Tokenizer and prepare datasets {'='*10}")

    #####################################
    # Load the SFT model
    #####################################
    logger.info(f"{'='*10} BEGIN Configure/Build the SFT Model {'='*10}")

    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map(),
        quantization_config=get_quantization_config(model_args),
    )

    model = model_args.model_name_or_path

    if is_adapter_model(model, model_args.model_revision):
        # load the model, merge the adapter weights and unload the adapter
        # Note: to run QLora, you will need to merge the based model separately as the merged model in 16bit
        logger.info(f"Merging peft adapters for {model_args.model_name_or_path=}")

        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)

        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map(),
        )

        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path, revision=model_args.model_revision)
        model.eval()
        model = model.merge_and_unload()
        model_kwargs = None

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None
    else:
        ref_model = model
        ref_model_kwargs = model_kwargs

    accelerator.wait_for_everyone()

    logger.info(f"{'='*10} END Configure/Build the SFT Model {'='*10}")

    #####################################
    # Instantiate DPO trainer
    #####################################
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
    )

    #####################################
    # Training loop
    #####################################
    logger.info(f"{'='*10} BEGIN Training {'='*10}")

    train_result = dpo_trainer.train()
    metrics = train_result.metrics

    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(raw_datasets["train"])
    metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))

    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()

    logger.info(f"{'='*10} END Training {'='*10}")

    #####################################
    # Evaluate
    #####################################
    if training_args.do_eval:
        logger.info(f"{'='*10} BEGIN Evaluation {'='*10}")
        metrics = dpo_trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["test"])
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["test"]))

        dpo_trainer.log_metrics("eval", metrics)
        dpo_trainer.save_metrics("eval", metrics)

        logger.info(f"{'='*10} END Evaluation {'='*10}")

    #####################################
    # Save model and create model card
    #####################################
    logger.info(f"{'='*10} BEGIN Save Model (Optional: Push to Hub) {'='*10}")

    dpo_trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to: {training_args.output_dir}")

    # Ensure we don't timeout on model save / push to Hub
    logger.info("Waiting for all processes to finish ...")
    accelerator.wait_for_everyone()
    accelerator.end_training()

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        dpo_trainer.create_model_card(**kwargs)

        # Restore k,v cache for fast inference
        dpo_trainer.model.config.use_cache = True
        dpo_trainer.model.config.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            dpo_trainer.push_to_hub()

    logger.info(f"{'='*10} END Save Model (Optional: Push to Hub) {'='*10}")
    logger.info(f"{'*'*10} END RUN {'*'*10}")


if __name__ == "__main__":
    main()
