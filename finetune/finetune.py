import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset, Dataset
import argparse
import inspect
from PIL import Image
import torch
from transformers import (
    BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor,
    get_scheduler, AdamW, AutoTokenizer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from ft_dataset import (
    Muitimodal_Dataset, Unimodal_Dataset,
    train_collate_fn_llava_muitimodal, train_collate_fn_llava_unimodal
)
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments
import random
from torch.utils.data import Subset
import glob

# Identify all linear layers in the model for applying LoRA

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Remove 'lm_head' if present (required for 16-bit precision)
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)

# Load model and processor based on the specified model ID

def load_model_and_processor(model_id):
    if model_id.startswith("llava"):
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


######################### Main Training Entry #################################
def main(args):
    print("Trainer Status is ", args.trainer)

    # Load model and processor
    model, processor = load_model_and_processor(args.model_id)
    print("Processor Tokenizer Length: ", len(processor.tokenizer))

    # Load tokenizer and ensure embedding size matches
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    os.makedirs(args.save_dir, exist_ok=True)

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("Getting PEFT model...")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Load dataset
    df = pd.read_parquet(args.data_dir)
    multimodel_dataset = Muitimodal_Dataset(df=df)
    unimodel_dataset = Unimodal_Dataset(df=df)

    # Build dataloaders for multimodal and unimodal training
    if args.model_id.startswith("llava"):
        train_dataloader_multimodal = DataLoader(
            multimodel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_muitimodal(x, processor, args)
        )
        train_dataloader_unimodal = DataLoader(
            unimodel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args)
        )

    # Initialize accelerator
    accelerator = Accelerator()

    # Optimizer and learning rate scheduler setup
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader_multimodal) * args.num_epochs,
    )

    # Prepare model and dataloaders with accelerator
    model, optimizer, train_dataloader_multimodal, train_dataloader_unimodal, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_multimodal, train_dataloader_unimodal, lr_scheduler
    )

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        multi_progress_bar = tqdm(train_dataloader_multimodal, desc=f"Epoch {epoch + 1}")

        for batch in multi_progress_bar:
            input_ids, attention_mask, pixel_values, labels = batch
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.item()

            multi_progress_bar.set_postfix(loss=total_loss / len(multi_progress_bar))
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader_multimodal)}")


        # uni_progress_bar = tqdm(train_dataloader_unimodal, desc=f"Epoch {epoch + 1}")
        # for batch in uni_progress_bar:
        #     input_ids, attention_mask, _, labels = batch
        #     outputs = model(input_ids=input_ids,
        #                     attention_mask=attention_mask,
        #                     labels=labels)
        #     loss = outputs.loss
        #     accelerator.backward(loss)
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     lr_scheduler.step()
        #     total_loss += loss.item()
        #     uni_progress_bar.set_postfix(loss=total_loss / len(uni_progress_bar))

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for configurable options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")

    args = parser.parse_args()
    main(args)
