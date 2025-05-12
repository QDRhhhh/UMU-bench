import pandas as pd
import copy
import json
from typing import Any, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
import os
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import DataLoader
import ast
import random

class Muitimodal_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame, mode='forget_5',target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        self.mode = mode
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # Extract the bytes from the 'image' dictionary
            image_data = row['image'].get('bytes')  # Access the image bytes

            # Convert the image bytes to a PIL Image
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue
            python_dict = ast.literal_eval(row['MM_QA'])
            json_str = json.dumps(python_dict, indent=4)
            QAs = json.loads(json_str)
            questions = QAs['question']
            answers = QAs['answer']
            for k in questions.keys():
                flattened_data.append({
                    "image": image,
                    "question":questions[k],
                    "answer": answers[k]
                })  
        if self.mode.split('_')[0]=='retain':
            ratio = int(self.mode.split('_')[1])/100
            n = int(len(flattened_data)*(1-ratio)/ratio)
            # print(ratio,n)
            random.seed(42)
            flattened_data = random.sample(flattened_data, n)

        return flattened_data
    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }

def train_collate_fn_llava_multimodal(examples, processor, args):
    images = []
    texts = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')
        images.append(image)
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        texts.append(prompt)

    if len(texts) == 0 or len(images) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")


    # Process the batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels


    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]



# from qwen_vl_utils import process_vision_info
# import base64
# from io import BytesIO


# def train_collate_fn_qwen2vl_multimodal(examples, processor, args):
#     texts = []
#     all_image_inputs = []
#     all_video_inputs = []

#     for example in examples:
#         image = example.get("image")
#         question = example.get("question", "")
#         answer = example.get("answer", "")
#         buffered = BytesIO()
#         image.save(buffered, format="JPEG")
#         img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#         # Create the conversation prompt with the image field filled with the actual image.
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": f"data:image/jpeg;base64,{img_base64}"},
#                     {"type": "text", "text": question}
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": answer}
#                 ]
#             }
#         ]
#         # For training, we typically do not add the generation prompt.
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#         texts.append(text.strip())

#         # Process vision info (images and videos) using Qwen2VL's utility.
#         image_inputs, video_inputs = process_vision_info(messages)
#         all_image_inputs.append(image_inputs)

#     if len(texts) == 0:
#         raise ValueError("Empty batch. No valid examples provided.")

#     # Prepare the batch with text, images, and videos.
#     batch = processor(
#         text=texts,
#         images=all_image_inputs,
#         # videos=all_video_inputs,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     )
#     # print(batch.keys())

#     # Prepare labels: first mask pad tokens with -100.
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     # print(processor.tokenizer.additional_special_tokens)
#     # Optionally, assign the image token ID to the appropriate positions.
#     # image_token_id = processor.tokenizer.additional_special_tokens_ids[
#     #     processor.tokenizer.additional_special_tokens.index("<image_pad>")
#     # ]
#     # labels[labels == processor.tokenizer.pad_token_id] = image_token_id

#     batch["labels"] = labels

#     if args.trainer:
#         return {
#             "input_ids": batch["input_ids"],
#             "attention_mask": batch["attention_mask"],
#             "pixel_values": batch.get("pixel_values"),
#             "labels": batch["labels"],
#             "image_grid_thw":batch['image_grid_thw']
#         }
#     else:
#         return (
#             batch["input_ids"],
#             batch["attention_mask"],
#             batch.get("pixel_values"),
#             batch["labels"],
#             batch['image_grid_thw']
#         )




class Unimodal_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame,mode='forget_5', target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        self.mode = mode
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # QAs = json.loads(row['UM_QA'])
            python_dict = ast.literal_eval(row['UM_QA'])
            json_str = json.dumps(python_dict, indent=4)
            QAs = json.loads(json_str)
            questions = QAs['question']
            answers = QAs['answer']
            for k in questions.keys():
                flattened_data.append({
                    "image": None,
                    "question":questions[k],
                    "answer": answers[k]
                })  
        if self.mode.split('_')[0]=='retain':
            ratio = int(self.mode.split('_')[1])/100
            n = int(len(flattened_data)*(1-ratio)/ratio)
            random.seed(42)
            flattened_data = random.sample(flattened_data, n)
        return flattened_data

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        # image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": None,
            "question": tokenized_question,
            "answer": tokenized_answer
        }

def train_collate_fn_llava_unimodal(examples, processor, args):
    texts = []
    for example in examples:
        question = example.get('question')
        answer = example.get('answer')
        prompt = f"USER: {question}\nASSISTANT: {answer}"
        texts.append(prompt)

    if len(texts) == 0:
        raise ValueError("Empty batch. No valid images or text in the examples provided.")


    # Process the batch
    batch = processor(
        text=texts,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt"
    )
    # Mask labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch["input_ids"], batch["attention_mask"], None, batch["labels"]



# def train_collate_fn_qwen2vl_unimodal(examples, processor, args):
#     texts = []
#     all_image_inputs = []
#     all_video_inputs = []

#     for example in examples:
#         question = example.get("question", "")
#         answer = example.get("answer", "")

#         # Create the conversation prompt with the image field filled with the actual image.
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": question}
#                 ]
#             },
#             {
#                 "role": "assistant",
#                 "content": [
#                     {"type": "text", "text": answer}
#                 ]
#             }
#         ]
#         # For training, we typically do not add the generation prompt.
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#         texts.append(text.strip())

#     if len(texts) == 0:
#         raise ValueError("Empty batch. No valid examples provided.")

#     # Prepare the batch with text, images, and videos.
#     batch = processor(
#         text=texts,
#         # images=all_image_inputs,
#         # videos=all_video_inputs,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     )
#     # print(batch.keys())

#     # Prepare labels: first mask pad tokens with -100.
#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100
#     # print(processor.tokenizer.additional_special_tokens)
#     # Optionally, assign the image token ID to the appropriate positions.
#     # image_token_id = processor.tokenizer.additional_special_tokens_ids[
#     #     processor.tokenizer.additional_special_tokens.index("<image_pad>")
#     # ]
#     # labels[labels == processor.tokenizer.pad_token_id] = image_token_id

#     batch["labels"] = labels

#     if args.trainer:
#         return {
#             "input_ids": batch["input_ids"],
#             "attention_mask": batch["attention_mask"],
#             "pixel_values": None,
#             "labels": batch["labels"],
#             "image_grid_thw":None
#         }
#     else:
#         return (
#             batch["input_ids"],
#             batch["attention_mask"],
#             None,
#             batch["labels"],
#             None
#         )