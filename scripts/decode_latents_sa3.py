import argparse
import json
import os

import torch
import torchaudio

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict


def find_first_latent(input_path):
    if os.path.isfile(input_path):
        return input_path

    latent_files = []
    for root, _, files in os.walk(input_path):
        for filename in files:
            if filename.endswith(".pt"):
                latent_files.append(os.path.join(root, filename))

    if not latent_files:
        raise FileNotFoundError(f"No .pt latent files found in {input_path}")

    return sorted(latent_files)[0]


def load_model(ckpt_dir, device, model_half):
    with open(os.path.join(ckpt_dir, "model_config.json")) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(os.path.join(ckpt_dir, "model.safetensors")))
    model.eval()
    model.requires_grad_(False)
    model = model.to(device)

    if model_half:
        model = model.to(torch.float16)

    return model, model_config


def prepare_latents(latents):
    if latents.ndim == 2:
        # encode_latents_sa3.py saves [T, C]; decode_audio expects [B, C, T].
        latents = latents.transpose(0, 1).unsqueeze(0)
    elif latents.ndim == 3:
        # Accept either [B, T, C] from saved batches or already channel-first [B, C, T].
        if latents.shape[1] > latents.shape[2]:
            latents = latents.transpose(1, 2)
    else:
        raise ValueError(f"Expected latent tensor with 2 or 3 dims, got shape {tuple(latents.shape)}")

    return latents.contiguous()


def main():
    parser = argparse.ArgumentParser(description="Decode SAME-L latents saved by encode_latents_sa3.py")
    parser.add_argument("--input", default="/cfs-r3ufsqcb/for_share/datasets/music_restoration_data/label_latents_stable_audio_3/582696488_012.pt", help="Latent .pt file or directory")
    parser.add_argument("--output", default="reconstructed.wav", help="Output wav path")
    parser.add_argument("--ckpt-dir", default="SAME-L", help="Local SAME-L checkpoint directory")
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 inference on CUDA")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_half = device == "cuda" and not args.fp32

    latent_path = find_first_latent(args.input)
    print(f"Loading latent: {latent_path}")

    model, model_config = load_model(args.ckpt_dir, device, model_half)
    sample_rate = model_config["sample_rate"]

    latents = torch.load(latent_path, map_location="cpu", weights_only=True)
    latents = prepare_latents(latents).to(device)
    if model_half:
        latents = latents.half()

    with torch.no_grad():
        reconstructed = model.decode_audio(latents)

    reconstructed = reconstructed.squeeze(0).detach().cpu().to(torch.float32).clamp(-1, 1)
    torchaudio.save(args.output, reconstructed, sample_rate)
    print(f"Saved reconstruction to {args.output}")


if __name__ == "__main__":
    main()
