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
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset, Audio
from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32


def normalize_audio(audio):
    """
    Convert any audio format to numpy waveform.
    Supports:
    - HuggingFace Audio dict
    - numpy arrays
    - file paths
    """

    if isinstance(audio, dict) and "array" in audio:
        return audio["array"]

    if isinstance(audio, np.ndarray):
        return audio

    if isinstance(audio, str):
        wav, _ = sf.read(audio)
        return wav

    raise TypeError(f"Unsupported audio format: {type(audio)}")


def load_data_from_source(input_source: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load dataset from JSONL or HuggingFace
    """

    # Local JSONL
    if Path(input_source).exists() and input_source.endswith(".jsonl"):

        print(f"Loading from local JSONL file: {input_source}")

        with open(input_source, "r") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]

        return data

    # HuggingFace dataset
    print(f"Loading from Hugging Face dataset: {input_source} (split: {split})")

    dataset = load_dataset(input_source, split=split)

    # ensure audio decoding
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    data = [dict(row) for row in dataset]

    print(f"Loaded {len(data)} samples from Hugging Face dataset")

    return data


def map_huggingface_fields(
    data: List[Dict[str, Any]], field_mapping: Dict[str, str] = None
) -> List[Dict[str, Any]]:

    if field_mapping is None:

        field_mapping = {
            "audio": "audio",
            "text": "text",
            "transcript": "text",
            "sentence": "text",
            "normalized_text": "text",
            "wav": "audio",
            "path": "audio",
        }

    mapped_data = []

    for item in data:

        mapped_item = {}

        # audio
        for hf_field, target in field_mapping.items():
            if target == "audio" and hf_field in item:
                mapped_item["audio"] = item[hf_field]
                break

        # text
        for hf_field, target in field_mapping.items():
            if target == "text" and hf_field in item:
                mapped_item["text"] = item[hf_field]
                break

        mapped_data.append(mapped_item)

    return mapped_data


def validate_data_format(data: List[Dict[str, Any]]):

    if not data:
        raise ValueError("Dataset empty")

    if "audio" not in data[0] or "text" not in data[0]:
        raise ValueError("Each sample must contain 'audio' and 'text'")


def prepare_data(
    input_source: str,
    output_jsonl: str,
    device: str,
    tokenizer_model_path: str,
    split: str = "train",
    field_mapping: str = None,
):

    total_lines = load_data_from_source(input_source, split)

    hf_field_mapping = None

    if field_mapping:
        hf_field_mapping = json.loads(field_mapping)

    total_lines = map_huggingface_fields(total_lines, hf_field_mapping)

    validate_data_format(total_lines)

    print(f"Loaded {len(total_lines)} samples")

    print("Loading tokenizer...")

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )

    final_lines = []
    batch_lines = []
    batch_audios = []

    print("Encoding audio files...")

    for idx, line in enumerate(total_lines):

        if idx % 10 == 0:
            print(f"Processing: {idx}/{len(total_lines)}")

        waveform = normalize_audio(line["audio"])

        batch_lines.append(line)
        batch_audios.append(waveform)

        if len(batch_lines) >= BATCH_INFER_NUM:

            enc_res = tokenizer.encode(batch_audios, sr=16000)

            for code, batch_line in zip(enc_res.audio_codes, batch_lines):

                batch_line["audio_codes"] = code.cpu().tolist()

                final_lines.append(batch_line)

            batch_lines.clear()
            batch_audios.clear()

    # remaining batch
    if batch_audios:

        enc_res = tokenizer.encode(batch_audios, sr=16000)

        for code, batch_line in zip(enc_res.audio_codes, batch_lines):

            batch_line["audio_codes"] = code.cpu().tolist()

            final_lines.append(batch_line)

    print(f"Writing {len(final_lines)} samples to {output_jsonl}")

    with open(output_jsonl, "w") as f:

        for line in final_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print("Data preparation complete!")


def main():

    parser = argparse.ArgumentParser(
        description="Prepare dataset for Qwen3-TTS finetuning"
    )

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--tokenizer_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    )

    parser.add_argument(
        "--input_source",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--field_mapping",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    prepare_data(
        input_source=args.input_source,
        output_jsonl=args.output_jsonl,
        device=args.device,
        tokenizer_model_path=args.tokenizer_model_path,
        split=args.split,
        field_mapping=args.field_mapping,
    )


if __name__ == "__main__":
    main()
