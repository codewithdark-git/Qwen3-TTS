# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset
from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

def load_data_from_source(input_source: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load data from either a local JSONL file or a Hugging Face dataset.
    
    Args:
        input_source: Either a path to a local JSONL file or a Hugging Face dataset ID
        split: Dataset split to load (default: "train")
    
    Returns:
        List of dictionaries containing the data
    """
    # Check if it's a local file
    if Path(input_source).exists() and input_source.endswith('.jsonl'):
        print(f"Loading from local JSONL file: {input_source}")
        with open(input_source, 'r') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    else:
        # Try to load from Hugging Face
        print(f"Loading from Hugging Face dataset: {input_source} (split: {split})")
        try:
            dataset = load_dataset(input_source, split=split)
            # Convert to list of dictionaries
            data = [dict(row) for row in dataset]
            print(f"Loaded {len(data)} samples from Hugging Face dataset")
            return data
        except Exception as e:
            raise ValueError(
                f"Failed to load data from '{input_source}'. "
                f"Please provide either a valid local JSONL file path or a Hugging Face dataset ID. "
                f"Error: {str(e)}"
            )

def map_huggingface_fields(data: List[Dict[str, Any]], field_mapping: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """
    Map Hugging Face dataset fields to the required format.
    
    Args:
        data: List of data dictionaries from Hugging Face
        field_mapping: Dictionary mapping HF field names to required names
                      e.g., {"audio_path": "audio", "transcript": "text"}
    
    Returns:
        List of dictionaries with mapped field names
    """
    if field_mapping is None:
        # Default mappings for common HF datasets
        field_mapping = {
            "audio": "audio",
            "text": "text",
            "transcript": "text",
            "sentence": "text",
            "normalized_text": "text",
            "wav": "audio",
            "path": "audio",
            "ref_audio": "ref_audio",
            "reference_audio": "ref_audio",
        }
    
    mapped_data = []
    for item in data:
        mapped_item = {}
        
        # Map audio field
        audio_found = False
        for hf_field, target_field in field_mapping.items():
            if target_field == "audio" and hf_field in item:
                mapped_item["audio"] = item[hf_field]
                audio_found = True
                break
        
        if not audio_found and "audio" in item:
            mapped_item["audio"] = item["audio"]
        
        # Map text field
        text_found = False
        for hf_field, target_field in field_mapping.items():
            if target_field == "text" and hf_field in item:
                mapped_item["text"] = item[hf_field]
                text_found = True
                break
        
        if not text_found and "text" in item:
            mapped_item["text"] = item["text"]
        
        # Map optional ref_audio field
        ref_audio_found = False
        for hf_field, target_field in field_mapping.items():
            if target_field == "ref_audio" and hf_field in item:
                mapped_item["ref_audio"] = item[hf_field]
                ref_audio_found = True
                break
        
        # If ref_audio not found, use audio as fallback
        if not ref_audio_found:
            mapped_item["ref_audio"] = mapped_item.get("audio", mapped_item.get("audio"))
        
        # Copy any additional fields that might be needed
        for key in item.keys():
            if key not in ["audio", "text", "transcript", "sentence", "wav", "path"]:
                if key not in mapped_item:
                    mapped_item[key] = item[key]
        
        mapped_data.append(mapped_item)
    
    return mapped_data

def validate_data_format(data: List[Dict[str, Any]]) -> None:
    """
    Validate that the data has the required fields.
    
    Args:
        data: List of data dictionaries
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = {"audio", "text"}
    
    if not data:
        raise ValueError("Data is empty")
    
    first_item = data[0]
    missing_fields = required_fields - set(first_item.keys())
    
    if missing_fields:
        raise ValueError(
            f"Data is missing required fields: {missing_fields}. "
            f"Each item must contain 'audio' and 'text'. "
            f"Available fields: {set(first_item.keys())}"
        )

def prepare_data(input_source: str, output_jsonl: str, device: str, 
                 tokenizer_model_path: str, split: str = "train", 
                 field_mapping: str = None) -> None:
    """
    Main function to prepare data by encoding audio files.
    
    Args:
        input_source: Path to local JSONL or Hugging Face dataset ID
        output_jsonl: Path to output JSONL file
        device: Device to use for tokenizer
        tokenizer_model_path: Path to the tokenizer model
        split: Dataset split to load (only for HF datasets)
        field_mapping: JSON string with field mappings for HF datasets
    """
    # Load data
    total_lines = load_data_from_source(input_source, split)
    
    # Parse field mapping if provided
    hf_field_mapping = None
    if field_mapping:
        try:
            hf_field_mapping = json.loads(field_mapping)
        except json.JSONDecodeError:
            print("Warning: Invalid field_mapping JSON. Using default mappings.")
    
    # Map HuggingFace fields to required format
    total_lines = map_huggingface_fields(total_lines, hf_field_mapping)
    
    # Validate format
    validate_data_format(total_lines)
    
    print(f"Loaded {len(total_lines)} samples")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )
    
    # Process data in batches
    final_lines = []
    batch_lines = []
    batch_audios = []
    
    print("Encoding audio files...")
    for idx, line in enumerate(total_lines):
        if idx % 10 == 0:
            print(f"Processing: {idx}/{len(total_lines)}")
        
        batch_lines.append(line)
        batch_audios.append(line['audio'])
        
        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, batch_line in zip(enc_res.audio_codes, batch_lines):
                batch_line['audio_codes'] = code.cpu().tolist()
                final_lines.append(batch_line)
            batch_lines.clear()
            batch_audios.clear()
    
    # Process remaining items
    if len(batch_audios) > 0:
        enc_res = tokenizer_12hz.encode(batch_audios)
        for code, batch_line in zip(enc_res.audio_codes, batch_lines):
            batch_line['audio_codes'] = code.cpu().tolist()
            final_lines.append(batch_line)
        batch_lines.clear()
        batch_audios.clear()
    
    # Write output
    print(f"Writing {len(final_lines)} samples to {output_jsonl}")
    final_lines_json = [json.dumps(line, ensure_ascii=False) for line in final_lines]
    
    with open(output_jsonl, 'w') as f:
        for line in final_lines_json:
            f.write(line + '\n')
    
    print("Data preparation complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for Qwen3-TTS fine-tuning from local JSONL or Hugging Face datasets"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="Device to use for encoding (default: cuda:0)"
    )
    parser.add_argument(
        "--tokenizer_model_path", 
        type=str, 
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Path to the tokenizer model"
    )
    parser.add_argument(
        "--input_source", 
        type=str, 
        required=True,
        help="Input source: either path to local JSONL file or Hugging Face dataset ID"
    )
    parser.add_argument(
        "--output_jsonl", 
        type=str, 
        required=True,
        help="Path to output JSONL file with audio codes"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train",
        help="Dataset split to load when using Hugging Face datasets (default: train)"
    )
    parser.add_argument(
        "--field_mapping",
        type=str,
        default=None,
        help='JSON string mapping HuggingFace field names to required names. '
             'Example: \'{"audio_path": "audio", "transcript": "text"}\''
    )
    
    args = parser.parse_args()
    
    try:
        prepare_data(
            input_source=args.input_source,
            output_jsonl=args.output_jsonl,
            device=args.device,
            tokenizer_model_path=args.tokenizer_model_path,
            split=args.split,
            field_mapping=args.field_mapping
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
