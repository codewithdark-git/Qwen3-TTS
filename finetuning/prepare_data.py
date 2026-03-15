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

BATCH_INFER_NUM = 16

# Directory where audio files are saved when the HuggingFace dataset provides
# raw waveform arrays instead of file paths.
AUDIO_CACHE_DIR = Path("audio_cache")


def normalize_audio(audio):
    """
    Convert any supported audio input to a 1-D numpy float32 waveform.
    Handles HuggingFace AudioDecoder, dicts, torch tensors, numpy arrays,
    and file paths.
    """
    import numpy as np

    # HuggingFace lazy AudioDecoder
    if hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        audio = decoded.data

    # HuggingFace dict format  {"array": ..., "sampling_rate": ...}
    if isinstance(audio, dict) and "array" in audio:
        audio = audio["array"]

    # File path
    elif isinstance(audio, str):
        audio, _ = sf.read(audio)

    # Torch tensor to numpy
    try:
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
    except Exception:
        pass

    # Convert lists / other iterables to numpy
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    audio = audio.astype(np.float32)

    # Squeeze extra dimensions, e.g. (1, N) -> (N,)
    audio = np.squeeze(audio)

    # Convert stereo/multi-channel to mono by averaging
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)

    if audio.ndim != 1:
        raise ValueError(
            f"Could not reduce audio to 1-D array, final shape={audio.shape}"
        )

    # Clip to valid float range for WAV
    audio = np.clip(audio, -1.0, 1.0)

    return audio


def get_audio_sr(audio_obj) -> int:
    """Extract sampling rate from a HuggingFace audio object or dict."""
    if isinstance(audio_obj, dict) and "sampling_rate" in audio_obj:
        return int(audio_obj["sampling_rate"])
    if hasattr(audio_obj, "sampling_rate"):
        return int(audio_obj.sampling_rate)
    # Fallback: we cast the dataset to Audio(sampling_rate=16000) earlier
    return 16000


def write_wav(path: str, waveform: np.ndarray, sr: int = 16000):
    """
    Write a 1-D float32 waveform to a WAV file.
    Uses scipy.io.wavfile (int16 PCM) as the primary method because it is
    more reliable than soundfile for arbitrary float32 arrays.
    Falls back to soundfile with explicit format flags if scipy is missing.
    """
    try:
        from scipy.io import wavfile as _wavfile
        pcm16 = (waveform * 32767.0).astype(np.int16)
        _wavfile.write(path, sr, pcm16)
    except ImportError:
        sf.write(path, waveform, samplerate=sr, format="WAV", subtype="PCM_16")


def resolve_audio_path(audio_obj, idx: int) -> str:
    """
    Return a filesystem path for the audio sample.

    If the dataset already has a real file path we use it directly.
    Otherwise the waveform is saved to AUDIO_CACHE_DIR/<idx>.wav so that
    dataset.py can load it with librosa. This guarantees the JSONL never
    contains a None path.
    """
    # HuggingFace dict that includes a real path
    if isinstance(audio_obj, dict):
        path_candidate = audio_obj.get("path") or audio_obj.get("file")
        if path_candidate and Path(str(path_candidate)).exists():
            return str(path_candidate)

    # Object with a .path attribute pointing to an existing file
    if hasattr(audio_obj, "path") and audio_obj.path and Path(audio_obj.path).exists():
        return str(audio_obj.path)

    # No usable path: save waveform to disk
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUDIO_CACHE_DIR / f"{idx:07d}.wav"

    if not out_path.exists():
        waveform = normalize_audio(audio_obj)
        sr = get_audio_sr(audio_obj)

        # Resample to 16 kHz to match the tokenizer expected input rate
        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        write_wav(str(out_path), waveform, sr)

    return str(out_path)


def load_data_from_source(input_source: str, split: str = "train") -> List[Dict[str, Any]]:
    """Load a dataset from a local JSONL file or a HuggingFace dataset ID."""

    if Path(input_source).exists() and input_source.endswith(".jsonl"):
        print(f"Loading from local JSONL file: {input_source}")
        with open(input_source, "r") as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data

    print(f"Loading from Hugging Face dataset: {input_source} (split: {split})")
    dataset = load_dataset(input_source, split=split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    data = [dict(row) for row in dataset]
    print(f"Loaded {len(data)} samples from Hugging Face dataset")
    return data


def map_huggingface_fields(
    data: List[Dict[str, Any]], field_mapping: Dict[str, str] = None
) -> List[Dict[str, Any]]:
    """Map HuggingFace dataset column names to the expected audio/text keys."""

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

        for hf_field, target in field_mapping.items():
            if target == "audio" and hf_field in item:
                mapped_item["audio"] = item[hf_field]
                break

        for hf_field, target in field_mapping.items():
            if target == "text" and hf_field in item:
                mapped_item["text"] = item[hf_field]
                break

        mapped_data.append(mapped_item)

    return mapped_data


def validate_data_format(data: List[Dict[str, Any]]):
    if not data:
        raise ValueError("Dataset is empty.")
    if "audio" not in data[0] or "text" not in data[0]:
        raise ValueError(
            "Each sample must contain 'audio' and 'text'. "
            f"First sample keys: {list(data[0].keys())}"
        )


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

        # Resolve (or cache) a real filesystem path before encoding
        audio_path = resolve_audio_path(line["audio"], idx)
        line["_resolved_audio_path"] = audio_path

        batch_lines.append(line)
        batch_audios.append(waveform)

        if len(batch_lines) >= BATCH_INFER_NUM:
            enc_res = tokenizer.encode(batch_audios, sr=16000)
            for code, batch_line in zip(enc_res.audio_codes, batch_lines):
                batch_line["audio_codes"] = code.cpu().tolist()
                final_lines.append(batch_line)
            batch_lines.clear()
            batch_audios.clear()

    # Process remaining samples
    if batch_audios:
        enc_res = tokenizer.encode(batch_audios, sr=16000)
        for code, batch_line in zip(enc_res.audio_codes, batch_lines):
            batch_line["audio_codes"] = code.cpu().tolist()
            final_lines.append(batch_line)

    print(f"Writing {len(final_lines)} samples to {output_jsonl}")

    with open(output_jsonl, "w") as f:
        for line in final_lines:
            audio_path = line["_resolved_audio_path"]
            output = {
                "text": line["text"],
                "audio": audio_path,
                "ref_audio": audio_path,
                "audio_codes": line["audio_codes"],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

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
    parser.add_argument("--input_source", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--field_mapping", type=str, default=None)

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
