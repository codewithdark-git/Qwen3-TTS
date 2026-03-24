#!/usr/bin/env python3
# coding=utf-8
"""
Inference script for the fine-tuned Qwen3-TTS Urdu model.

Usage examples:

  # Fine-tuned checkpoint (local directory)
  python inference.py \
      --checkpoint "output_model/checkpoint-epoch-2" \
      --text "آپ کا استقبال ہے" \
      --speaker "speaker_test" \
      --output "urdu_output.wav"

  # Base model via HF cache snapshot path
  python inference.py \
      --checkpoint "/home/Ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3" \
      --text "آپ کا استقبال ہے" \
      --output "base_output.wav"

  # Base model via HF hub ID (downloads/uses cache automatically)
  python inference.py \
      --checkpoint "Qwen/Qwen3-TTS-12Hz-1.7B-Base" \
      --text "آپ کا استقبال ہے" \
      --output "base_output.wav"
"""

import argparse
import json
import os

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def is_local_path(path: str) -> bool:
    """
    Return True if path should be treated as a local filesystem path.

    Covers:
      - Absolute paths          : /home/Ubuntu/output_model/checkpoint-epoch-2
      - Relative paths          : output_model/checkpoint-epoch-2  or  ./ckpt
      - Paths that already exist on disk (absolute or relative)
      - Anything containing os.sep that isn't a HF repo ID (no slash = hub ID)

    A HuggingFace hub ID looks like "Qwen/Qwen3-TTS-12Hz-1.7B-Base" —
    exactly one forward slash, no other path indicators.
    """
    # Explicit absolute path
    if os.path.isabs(path):
        return True

    # Already exists as a relative path from cwd
    if os.path.exists(path):
        return True

    # Starts with ./ or ../ — clearly a relative filesystem path
    if path.startswith("./") or path.startswith("../"):
        return True

    # Contains OS path separator beyond a single slash (e.g. a/b/c)
    # HF repo IDs have exactly one slash: "org/repo"
    parts = path.replace("\\", "/").split("/")
    if len(parts) > 2:
        return True

    # Single-slash string like "output_model/checkpoint-epoch-2":
    # if neither part looks like a valid HF org/repo name it's a local path.
    # HF org names never contain hyphens followed by digits like "epoch-2",
    # but rather than heuristics we just check if it exists when resolved.
    if len(parts) == 2:
        # If the first component exists as a directory in cwd, it's local
        if os.path.isdir(parts[0]):
            return True

    return False


def resolve_model_path(path: str) -> str:
    """
    Return a guaranteed-local directory path.

    - Local paths are returned as-is (absolute) after existence check.
    - HF hub IDs are resolved from cache or downloaded.
    """
    if is_local_path(path):
        # Convert to absolute so callers never have cwd-dependent bugs
        abs_path = os.path.abspath(path)
        if not os.path.isdir(abs_path):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {abs_path}\n"
                "Make sure training has completed and the path is correct."
            )
        return abs_path

    # HuggingFace hub ID — resolve from local cache first, download if needed
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        try:
            local = snapshot_download(repo_id=path, local_files_only=True)
            print(f"Resolved '{path}' from HF cache: {local}")
            return local
        except LocalEntryNotFoundError:
            print(f"Downloading '{path}' from HuggingFace Hub...")
            local = snapshot_download(repo_id=path)
            print(f"Downloaded to: {local}")
            return local
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required for HF hub IDs. "
            "Install with: pip install huggingface_hub"
        )


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Local checkpoint directory (e.g. output_model/checkpoint-epoch-2), "
            "absolute HF cache snapshot path, or a HuggingFace hub ID."
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesise.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="speaker_test",
        help="Speaker name used during fine-tuning (default: speaker_test).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda:0 if available, else cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16).",
    )
    args = parser.parse_args()

    # Resolve checkpoint to a real local directory
    checkpoint_path = resolve_model_path(args.checkpoint)
    print(f"Loading model from: {checkpoint_path}")

    # dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load model
    load_kwargs = dict(device_map=args.device, dtype=dtype)
    # if args.device.startswith("cuda") and args.dtype in ("bfloat16", "float16"):
    #     load_kwargs["attn_implementation"] = "flash_attention_2"

    tts = Qwen3TTSModel.from_pretrained(checkpoint_path, **load_kwargs)
    print("Model loaded.")

    # Detect fine-tuned custom voice vs base model
    config_path = os.path.join(checkpoint_path, "config.json")
    is_custom_voice = False
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        is_custom_voice = cfg.get("tts_model_type") == "custom_voice"

    # Generate
    print(f"Synthesising: {args.text!r}")
    if is_custom_voice:
        print(f"Custom voice speaker: {args.speaker!r}")
        wavs, sr = tts.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
        )
    else:
        print("Base model — using standard generate().")
        wavs, sr = tts.generate(text=args.text)

    # Save
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(args.output, wavs[0], sr)
    print(f"Saved: {args.output}  (sample_rate={sr})")


if __name__ == "__main__":
    main()
