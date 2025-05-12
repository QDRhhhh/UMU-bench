import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration
import pandas as pd
import random
import json
from PIL import Image
from io import BytesIO
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import ast
import sys
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import difflib

def load_and_combine_parquet_files(directory):
    # Get all Parquet files in the directory
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

    # Read and concatenate all Parquet files
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df


def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score


def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt

def select_answer(assistant_response, temp):
    # 使用 difflib 来计算字符串的相似度
    similarities = [difflib.SequenceMatcher(None, assistant_response, t).ratio() for t in temp]
    
    # 找到最相似的字符串的索引
    most_similar_index = similarities.index(max(similarities))
    
    # 返回最相似的字符串
    return temp[most_similar_index]


def evaluate_classification(parquet_file,  processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################" )

    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the main Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0
    unimodal = []
    multimodal = []

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    # Process each evaluation sample
    for _, row in tqdm(eval_samples.iterrows()):
        python_dict = ast.literal_eval(row["Classify"])
        json_str = json.dumps(python_dict, indent=4)
        classification_questions = json.loads(json_str)
        image_data = row['image'].get('bytes')  # Access the image bytes
        image = Image.open(BytesIO(image_data)).convert("RGB")
        uni = classification_questions['unimodal']
        mul = classification_questions['muitimodal']
        keys = list(uni.keys())
        # Iterate through each image-textual question
        # print("########################## Processing Image-Textual Questions ########################## ")
        for key in keys:
            question = mul[key]['question']
            options =  mul[key]['options']
            temp = []
            temp.append(options['A'])
            temp.append(options['B'])
            temp.append(options['C'])
            temp.append(options['D'])
            temp_str = str(temp)
            # correct_answer = mul[key]['answer'].split('.')[0]
            correct_answer = options[uni[key]['answer'].split('.')[0]]
            question_with_options = formulate_prompt_with_options(question, options)


            # prompt = (f"USER: <image>\n{question_with_options}\n"
            #           f"Just give ONE letter representing the answer directly.\nASSISTANT:")


            if args.model_id.startswith("llava"):
                prompt = (f"USER: <image>\n{question}\nSelect answer in {temp_str}\n"
                      f"ASSISTANT:")
                inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
                assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
                if mode == 'default':
                    predicted_answer = assistant_response
                else:
                    predicted_answer = select_answer(assistant_response,temp)
            elif args.model_id.startswith("Qwen"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": f"{question}\nSelect answer in {temp_str}"},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
                predicted_answer = select_answer(generated_text,temp)
            # predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            
            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
                multimodal.append(1)
            else:
                multimodal.append(0)
            total_image_textual_questions += 1
            # print("Prompt: ", prompt)
            # print("Model Answer: ", predicted_answer)
            # print('generate: ',generated_text)
            # print("Correct Answer: ", correct_answer)
            # print("The model answer is: ", predicted_answer == correct_answer)
            # print("\n")

        # Process Pure_Text_Questions
        # print("########################## Processing Pure-textual Questions ########################## ")
        for key in keys:
            question = uni[key]['question']
            options =  uni[key]['options']
            # correct_answer = uni[key]['answer'].split('.')[0]
            correct_answer = options[uni[key]['answer'].split('.')[0]]
            temp = []
            temp.append(options['A'])
            temp.append(options['B'])
            temp.append(options['C'])
            temp.append(options['D'])
            temp_str = str(temp)
            question_with_options = formulate_prompt_with_options(question, options)

            # prompt = (
            #     f"USER:\n{question_with_options}\n"
            #     f"Just give ONE letter representing the answer directly.\nASSISTANT:"
            # )


            if args.model_id.startswith("llava"):
                prompt = (
                    f"USER:\n{question}\nSelect answer in {temp_str}\n"
                    f"ASSISTANT:"
                )
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)
                assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
                if mode == 'default':
                    predicted_answer = assistant_response
                else:
                    predicted_answer = select_answer(assistant_response,temp)
            elif args.model_id.startswith("Qwen"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{question}\nSelect answer in {temp_str}"},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0]
                predicted_answer = select_answer(generated_text,temp)
            # predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            
            if predicted_answer == correct_answer:
                total_pure_text_correct += 1
                unimodal.append(1)
            else:
                unimodal.append(0)
            total_pure_text_questions += 1

            # print("Prompt: ", prompt)
            # print("Model Answer: ", predicted_answer)
            # print('generate: ',generated_text)
            # print("Correct Answer: ", correct_answer)
            # print("The model answer is: ", predicted_answer == correct_answer)
            # print("\n")

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0
    all_modal_accuracy = 0
    for uni,muiti in zip(unimodal,multimodal):
        if uni==1 and muiti==1:
            all_modal_accuracy += 1
    all_modal_accuracy = all_modal_accuracy*100/len(unimodal)
    all_modal_error = 0
    for uni,muiti in zip(unimodal,multimodal):
        if uni==0 and muiti==0:
            all_modal_error += 1
    all_modal_error = all_modal_error*100/len(unimodal)
    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")
    print(f"All Modal Question Accuracy: {all_modal_accuracy:.2f}%")
    print(f"All Modal Question Error: {all_modal_error:.2f}%")
    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy,
        "All Modal Question Accuracy": all_modal_accuracy,
        "All Modal Question Error": all_modal_error
    }


# def evaluate_fill_in_the_blank(json_files, image_folder, processor, tokenizer, model, args, id_list_file=None, mode="default"):
def evaluate_fill_in_the_blank(parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    print(
        "################################## Fill-in-the-blank Task Starts ##############################################")

    print(f"Evaluating {mode} Mode")
    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0
    unimodal = []
    multimodal = []

    # id_list = id_list[0:2]
    # Load evaluation samples
    # Load the test set with multiple Parquet files if mode is "test"
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in tqdm(eval_samples.iterrows()):
        python_dict = ast.literal_eval(row["Cloze"])
        json_str = json.dumps(python_dict, indent=4)
        fill_in_the_blank_questions = json.loads(json_str)
        image_data = row['image'].get('bytes')  # Access the image bytes
        image = Image.open(BytesIO(image_data)).convert("RGB")
        uni = fill_in_the_blank_questions['unimodal']
        mul = fill_in_the_blank_questions['muitimodal']
        keys = list(uni.keys())
        # multimodal
        for key in keys:
            question = mul[key]["question"]
            ground_truth = mul[key]["answer"].split('.')[0]
            question = question + "\nPlease **ONLY** provide the correct answer without any explanation"
            if args.model_id.startswith("llava"):
                prompt = (f"USER: <image>\n{question}\nASSISTANT:")
                inputs = processor(images=[image],text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
                if "ASSISTANT:" in generated_text:
                    assistant_response = generated_text.split("ASSISTANT:")[1].strip()
                elif "Answer:" in generated_text:
                    assistant_response = generated_text.split("Answer:")[1].strip()
                else:
                    assistant_response = generated_text.strip()
            elif args.model_id.startswith("Qwen"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": f"{question}"},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                assistant_response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0].split('.')[0]
            # Process the answer

            # print("Prompt: ", prompt)
            # print("Model Answer: ", assistant_response)
            # print("Correct Answer: ", ground_truth)
            # print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            # print("\n")
            # Evaluate if the generated answer contains the correct ground truth
            if ground_truth.lower() in assistant_response.lower():
                total_image_textual_correct += 1
                multimodal.append(1)
            else:
                multimodal.append(0)
            total_image_textual_questions += 1
        # unimodal
        for key in keys:
            question = uni[key]["question"]
            ground_truth = uni[key]["answer"].split('.')[0]
            question = question + "\nPlease **ONLY** provide the correct answer without any explanation"
            if args.model_id.startswith("llava"):
                prompt = (f"USER: {question}\nASSISTANT:")
                inputs = processor(text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
                # Process the answer
                if "ASSISTANT:" in generated_text:
                    assistant_response = generated_text.split("ASSISTANT:")[1].strip()
                elif "Answer:" in generated_text:
                    assistant_response = generated_text.split("Answer:")[1].strip()
                else:
                    assistant_response = generated_text.strip()
            elif args.model_id.startswith("Qwen"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{question}"},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                assistant_response = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0].split('.')[0]

            # print("Prompt: ", prompt)
            # print("Model Answer: ", assistant_response)
            # print("Correct Answer: ", ground_truth)
            # print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            # print("\n")
            # Evaluate if the generated answer contains the correct ground truth
            if ground_truth.lower() in assistant_response.lower():
                total_pure_text_correct += 1
                unimodal.append(1)
            else:
                unimodal.append(0)
            total_pure_text_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0
    
    all_modal_accuracy = 0
    for uni,muiti in zip(unimodal,multimodal):
        if uni==1 and muiti==1:
            all_modal_accuracy += 1
    all_modal_accuracy = all_modal_accuracy*100/len(unimodal)
    all_modal_error = 0
    for uni,muiti in zip(unimodal,multimodal):
        if uni==0 and muiti==0:
            all_modal_error += 1
    all_modal_error = all_modal_error*100/len(unimodal)

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")
    print(f"All Modal Question Accuracy: {all_modal_accuracy:.2f}%")
    print(f"All Modal Question Error: {all_modal_error:.2f}%")

    return {
        "image_textual_accuracy": image_textual_accuracy,
        "pure_text_accuracy": pure_text_accuracy,
        "All Modal Question Accuracy": all_modal_accuracy,
        "All Modal Question Error":all_modal_error
    }

def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None):

    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    # Load the ID list from the forget Parquet file for filtering if mode is "test"
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # Load all IDs from the Parquet file if no filtering is needed
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()
    multimodal = []
    unimodal = []
    # id_list = id_list[0:2]
    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Loop through each person's data in the evaluation samples
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        python_dict = ast.literal_eval(row["Generation"])
        json_str = json.dumps(python_dict, indent=4)
        generation_questions = json.loads(json_str)
        image_data = row['image'].get('bytes')  # Access the image bytes
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_id = row['ID']
        uni = generation_questions['unimodal']
        mul = generation_questions['muitimodal']
        keys = list(uni.keys())

        # Process each generation question
        for key in keys:
            question_type = 'multimodal'
            question = mul[key]["question"]
            ground_truth = mul[key]["answer"]
            
            if args.model_id.startswith("llava"):
                prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.\nASSISTANT: "
                inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)
                if "ASSISTANT:" in generated_answer:
                    predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
                elif "Answer:" in generated_answer:
                    predicted_answer = generated_answer.split("Answer:")[1].strip()
                else:
                    predicted_answer = generated_answer.strip()
            elif args.model_id.startswith("Qwen"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH."},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                predicted_answer = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0].split('.')[0]

            # # Print debug information
            # print("###### Generation Question: ######", question)
            # print("###### Generation Prompt: ######", prompt)
            # print("###### Generation ASSISTANT: ######", predicted_answer)
            # print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)
            total_bleu_img += bleu_score
            total_rouge1_img += rouge_scores['rouge1'].fmeasure
            total_rouge2_img += rouge_scores['rouge2'].fmeasure
            total_rougeL_img += rouge_scores['rougeL'].fmeasure
            multimodal.append(rouge_scores['rougeL'].fmeasure)
            total_image_textual_questions += 1

        for key in keys:
            question_type = 'unimodal'
            question = uni[key]["question"]
            ground_truth = uni[key]["answer"]
            if args.model_id.startswith("llava"):
                prompt = f"USER: {question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH.\nASSISTANT:"
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
                generated_answer = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)
                # Process the generated answer
                if "ASSISTANT:" in generated_answer:
                    predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
                elif "Answer:" in generated_answer:
                    predicted_answer = generated_answer.split("Answer:")[1].strip()
                else:
                    predicted_answer = generated_answer.strip()
            elif args.model_id.startswith("Qwen"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH."},
                        ],
                    }
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                predicted_answer = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0].split('.')[0]
            # # Print debug information
            # print("###### Generation Question: ######", question)
            # print("###### Generation Prompt: ######", prompt)
            # print("###### Generation ASSISTANT: ######", predicted_answer)
            # print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            total_bleu_text += bleu_score
            total_rouge1_text += rouge_scores['rouge1'].fmeasure
            total_rouge2_text += rouge_scores['rouge2'].fmeasure
            total_rougeL_text += rouge_scores['rougeL'].fmeasure
            unimodal.append(rouge_scores['rougeL'].fmeasure)
            total_pure_text_questions += 1
    # # Save the results to a JSON file
    # if not os.path.exists(args.output_folder):
    #     os.makedirs(args.output_folder)

    # with open(f'{args.output_folder}/{mode}_generation_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)
    if mode=='forget':
        H = [(a*a+ b*b) / (a + b) if (a + b) != 0 else 0 for a, b in zip(multimodal, unimodal)]
        all_modal_RL = sum(H) / len(H)
    else:
        H = [(2*a*b) / (a + b) if (a + b) != 0 else 0 for a, b in zip(multimodal, unimodal)]
        all_modal_RL = sum(H) / len(H)
    # Calculate and print average ROUGE and BLEU scores
    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions,
            "ALL ROUGE-L (Image_Textual)":multimodal
        })

    if total_pure_text_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
            "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
            "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
            "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions,
            "ALL ROUGE-L (Pure_Text)": unimodal
        })
    avg_scores.update({"All Modal Average ROUGE-L": all_modal_RL})
    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument("--model_id", type=str, default='llava-hf/llava-1.5-7b-hf', help="Pretrained model ID")
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--forget_ratio', type=int, required=True, help='Path to real person image folder.')
    parser.add_argument("--data_split_dir", type=str, required=True, help="Directory of the test dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Directory of output")
    parser.add_argument("--output_file", type=str, required=True, help="Name of output file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_ratio}")
    real_folder = os.path.join(args.data_split_dir, "real_person")

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    real_parquet_file = os.path.join(real_folder, f"train-00000-of-00001.parquet")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()

    if args.model_id.startswith("llava"):
        print("Loading LLAVA Vanilla model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")


    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    print("### Evaluating Forget Set ###")
    forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")
    
    forget_classification_result = evaluate_classification(parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="default")
    forget_generation_result = evaluate_generation(parquet_file=forget_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="forget")
    print("### Evaluating Retain Shared Set ###")
    retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=retain_parquet_file,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_shared")

    retain_classification_result = evaluate_classification(parquet_file=retain_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="default")

    retain_generation_result = evaluate_generation(parquet_file=retain_parquet_file,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_shared")
    print("### Evaluating Real Person Set ###")
    real_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=real_parquet_file,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_shared")

    real_classification_result = evaluate_classification(parquet_file=real_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="real_person")

    real_generation_result = evaluate_generation(parquet_file=real_parquet_file,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_shared")

    results_data = {
        "Forget Results": {
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result
        },
        "Retain Results": {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result
        },
        "Real Person Results": {
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result
        }
    }
    os.makedirs(args.output_path, exist_ok=True)

    # 拼接完整输出路径
    full_output_path = os.path.join(args.output_path, args.output_file)
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()


