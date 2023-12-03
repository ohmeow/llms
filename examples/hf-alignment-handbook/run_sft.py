import logging, os, random, sys
from datetime import datetime

import datasets
import numpy as np
import torch
import transformers

from accelerate import Accelerator
from datasets import load_metric
from safetensors.torch import load_model, save_model
from transformers import set_seed
from transformers.integrations import WandbCallback
from trl import SFTTrainer
import wandb

from training_lib.configs import DataArguments, H4ArgumentParser, ModelArguments, SFTConfig
from training_lib.data import get_datasets
from training_lib.model_utils import apply_chat_template, get_kbit_device_map, get_tokenizer, get_peft_config, get_quantization_config


logger = logging.getLogger(__name__)


def main():
    logger.info(f"{'='*10} BEGIN RUN {'='*10}")

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
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

        wandb_tags = ["sft"]
        if is_testing:
            wandb_tags.append("test")

        accelerator.init_trackers(
            "zephyr-7b",
            config=model_args,
            init_kwargs={
                "wandb": {
                    "name": f"zephyr-7b-sft-{'lora' if use_peft else 'full'}-{'test' if is_testing else mmmmyydd}",
                    "job_type": "training",
                    "notes": "HuggingFace Alignment Recepipe: zephyr-7b-B SFT training script",
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
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.info(f"{'='*10} CONFIGURATION {'='*10}")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args.__dict__}")
    logger.info(f"Data parameters {data_args.__dict__}")
    logger.info(f"Training/evaluation parameters {training_args.__dict__}")

    #####################################
    # Load Tokenizer and prepare datasets
    #####################################
    logger.info(f"{'='*10} BEGIN Load Tokenizer and prepare datasets {'='*10}")

    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    # Grab the tokenizer
    tokenizer = get_tokenizer(model_args, data_args)

    # Apply the chat template
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    logger.info(f"{'='*10} END Load Tokenizer and prepare datasets {'='*10}")

    #####################################
    # Load Pretrained model
    #####################################
    logger.info(f"{'='*10} BEGIN Configure Model {'='*10}")

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

    logger.info(f"{'='*10} END Configure Model {'='*10}")

    #####################################
    # Initialize the Trainer
    #####################################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
    )

    #####################################
    # Training loop
    #####################################
    logger.info(f"{'='*10} BEGIN Training {'='*10}")

    train_result = trainer.train()
    metrics = train_result.metrics

    max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info(f"{'='*10} END Training {'='*10}")

    #####################################
    # Evaluate
    #####################################
    if training_args.do_eval:
        logger.info(f"{'='*10} BEGIN Evaluation {'='*10}")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        logger.info(f"{'='*10} END Evaluation {'='*10}")

    #####################################
    # Save model and create model card
    #####################################
    logger.info(f"{'='*10} BEGIN Save Model (Optional: Push to Hub) {'='*10}")

    # save_model(trainer.model, training_args.output_dir) if model_args.use_peft else trainer.save_model(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
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
        trainer.create_model_card(**kwargs)

        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        if training_args.push_to_hub is True:
            logger.info("Pushing to hub...")
            trainer.push_to_hub()

    logger.info(f"{'='*10} END Save Model (Optional: Push to Hub) {'='*10}")
    logger.info(f"{'*'*10} END RUN {'*'*10}")


if __name__ == "__main__":
    main()
