import pandas as pd
import json
import ast
from copy import deepcopy
from typing import List, Dict
import argparse
import os


def process_qa_data(
    parquet_path: str,
    image_dir: str,
    output_json_path: str
) -> None:
    template = {
        "messages": [
            {"content": "<image>", "role": "user"},
            {"content": "", "role": "assistant"}
        ],
        "images": [""]
    }

    df = pd.read_parquet(parquet_path)
    res: List[Dict] = []

    def extract_qa(column: str):
        for _, row in df.iterrows():
            try:
                qa_data = ast.literal_eval(row[column])
                questions = qa_data.get('question', {})
                answers = qa_data.get('answer', {})
                image_path = os.path.join(image_dir, f"image_{row['ID']}.jpg")

                for k in questions:
                    item = deepcopy(template)
                    item['messages'][0]['content'] += questions[k]
                    item['messages'][1]['content'] += answers.get(k, "")
                    item['images'][0] = image_path
                    res.append(item)
            except Exception as e:
                print(f"Error processing row ID {row.get('ID', 'Unknown')}: {e}")

    extract_qa('MM_QA')
    extract_qa('UM_QA')

    print(f"Total entries: {len(res)}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Process QA data and generate JSON.")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the input parquet file.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing image files.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output JSON file.")

    args = parser.parse_args()

    process_qa_data(
        parquet_path=args.parquet_path,
        image_dir=args.image_dir,
        output_json_path=args.output_json
    )


if __name__ == "__main__":
    main()
