import os
import sys
import json
import random
from collections import defaultdict, Counter
import argparse
# System path adjustments
sys.path.append('../')
sys.path.append('../../')

# Third-party libraries
import pandas as pd
from tqdm import tqdm
from PIL import Image

# PyTorch & related libraries
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim import AdamW
from accelerate import Accelerator

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    get_scheduler,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

# PEFT (Parameter-Efficient Fine-Tuning)
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

# Hugging Face Datasets
from datasets import load_dataset, Dataset

# Custom modules
from unlearn_dataset import (
    Muitimodal_Dataset,
    Unimodal_Dataset,
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    # print(list(lora_module_names))
    # sys.exit(0)
    return list(lora_module_names)


# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if args.model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Additional processor configuration if necessary
    processor.tokenizer.padding_side = "right"  # Ensure right padding
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    return model, processor


def invoke(batch,model,model_id,mode):
    if model_id.startswith("llava"):
        if mode == 'multimodal':
            input_ids, attention_mask, pixel_values, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )
        else:
            input_ids, attention_mask, _, labels = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")
    return outputs


######################### Accelerate Version #################################
def main(args):
    # Load model and processor

    model, processor = load_model_and_processor(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # LoRA configuration
    lora_config = LoraConfig(
        r=64, #32
        lora_alpha=32, #8
        lora_dropout=0.05,
        target_modules=['o_proj', 'q_proj', 'k_proj', 'down_proj', 'v_proj', 'up_proj', 'gate_proj'],
        # target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup

    # dataset = Vanilla_LLaVA_Dataset_baseline(json_dir=profile_dir, image_dir=image_base_path, flatten=False)
    # print(f"Dataset size (profiles): {len(dataset)}")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    df = pd.read_parquet(forget_parquet_file)

    multimodel_dataset = Muitimodal_Dataset(df=df)
    unimodel_dataset = Unimodal_Dataset(df=df)

    if args.model_id.startswith("llava"):
        train_dataloader_multimodal = DataLoader(
            multimodel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_multimodal(x, processor, args)
        )
        train_dataloader_unimodal = DataLoader(
            unimodel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader_multimodal) * args.num_epochs,
    )

    model, optimizer, train_dataloader_multimodal,train_dataloader_unimodal, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_multimodal,train_dataloader_unimodal, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        mix_progress_bar = tqdm(zip(train_dataloader_multimodal, train_dataloader_unimodal),
                                desc=f"Epoch {epoch + 1}",
                                total=len(train_dataloader_multimodal))  # 或者用 len(train_dataloader_unimodal)
        for multi_batch, uni_batch in mix_progress_bar:
            outputs = invoke(multi_batch,model,args.model_id,'multimodal')
            # ------------------- 多模态 forward + backward ------------------- 
            loss_multi = -outputs.loss
            accelerator.backward(loss_multi)
            # ------------------- 单模态 forward + backward -------------------
            outputs = invoke(uni_batch,model,args.model_id,'unimodal')
            loss_uni = -1*args.alpha*outputs.loss
            accelerator.backward(loss_uni)
            # accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            step_loss = loss_multi.item() + loss_uni.item()
            total_loss += step_loss
            mix_progress_bar.set_postfix({"step_loss": step_loss, "total_loss": total_loss})

        # 如果需要每个epoch结束时打印一下平均loss，可以加在循环外
        avg_loss = total_loss / (len(train_dataloader_multimodal))
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # if args.model_id.startswith("meta-llama") == False:
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, default='llava-hf/llava-1.5-7b-hf', help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Model path")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--data_split_dir", type=str, required=True, help="Directory of the test dataset")
    parser.add_argument("--forget_split_ratio", type=int, default=5, help="forget ratio")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training")
    parser.add_argument("--alpha", type=float, default=1, help="alpha")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")

    args = parser.parse_args()

    # Call main function
    main(args)
