#!/usr/bin/env python3
# coding=utf-8
"""
Inference script for the fine-tuned Qwen3-TTS Urdu model.

Usage:
    python inference.py \
        --checkpoint "output_model/checkpoint-epoch-2" \
        --text "آپ کا استقبال ہے" \
        --output "output.wav"

    # or use the HF cache path directly:
    python inference.py \
        --checkpoint "/home/Ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3" \
        --text "آپ کا استقبال ہے" \
        --speaker "speaker_test" \
        --output "output.wav"
"""

import argparse
import os

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


def resolve_model_path(model_path: str) -> str:
    """
    Accept either:
      - A local directory path  (e.g. "output_model/checkpoint-epoch-2")
      - A HuggingFace hub ID   (e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    Returns a guaranteed-local path in both cases.
    """
    if os.path.isdir(model_path):
        return model_path

    # HF hub ID — resolve from cache or download
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError
        try:
            local = snapshot_download(repo_id=model_path, local_files_only=True)
            print(f"Resolved '{model_path}' from HF cache: {local}")
            return local
        except LocalEntryNotFoundError:
            print(f"Downloading '{model_path}' from HuggingFace Hub...")
            local = snapshot_download(repo_id=model_path)
            print(f"Downloaded to: {local}")
            return local
    except ImportError:
        raise RuntimeError("pip install huggingface_hub")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to a fine-tuned checkpoint directory "
            "(e.g. output_model/checkpoint-epoch-2) "
            "OR the HF cache snapshot path "
            "OR a HuggingFace hub ID."
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesise (Urdu or any language the model supports).",
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
        help="Path to write the output WAV file (default: output.wav).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 if available).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for the model (default: bfloat16).",
    )
    args = parser.parse_args()

    # ── resolve checkpoint path ───────────────────────────────────────────
    checkpoint_path = resolve_model_path(args.checkpoint)
    print(f"Loading model from: {checkpoint_path}")

    # ── dtype ─────────────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ── load model ────────────────────────────────────────────────────────
    load_kwargs = dict(
        device_map=args.device,
        dtype=dtype,
    )
    # flash_attention_2 only works on CUDA with bfloat16/float16
    if args.device.startswith("cuda") and args.dtype in ("bfloat16", "float16"):
        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    tts = Qwen3TTSModel.from_pretrained(checkpoint_path, **load_kwargs)
    print("Model loaded.")

    # ── detect model type (fine-tuned custom voice vs base) ───────────────
    import json
    config_path = os.path.join(checkpoint_path, "config.json")
    is_custom_voice = False
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        is_custom_voice = cfg.get("tts_model_type") == "custom_voice"

    # ── generate ──────────────────────────────────────────────────────────
    print(f"Synthesising: {args.text!r}")
    if is_custom_voice:
        print(f"Using custom voice speaker: {args.speaker!r}")
        wavs, sr = tts.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
        )
    else:
        # Base model — standard generation (no custom speaker)
        print("Base model detected — using standard generate().")
        wavs, sr = tts.generate(text=args.text)

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, wavs[0], sr)
    print(f"Saved to: {args.output}  (sample_rate={sr})")


if __name__ == "__main__":
    main()
