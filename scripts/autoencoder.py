import json
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

device = "cuda" if torch.cuda.is_available() else "cpu"
model_half = device == "cuda"

ckpt_dir = "SAME-L"

with open(f"{ckpt_dir}/model_config.json") as f:
    model_config = json.load(f)

model = create_model_from_config(model_config)
model.load_state_dict(load_ckpt_state_dict(f"{ckpt_dir}/model.safetensors"))
model.eval()

sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)
if model_half:
    model = model.to(torch.float16)
audio_file = '/cfs-r3ufsqcb/for_share/datasets/music_restoration_data/test_data_3/213221098.flac'
audio, sr = torchaudio.load(audio_file)
if audio.shape[0] == 1:
    audio = audio.repeat(2, 1)

audio = audio.unsqueeze(0).to(device)
if model_half:
  audio = audio.half()
with torch.no_grad():
    latents = model.encode_audio(audio)  
    reconstructed = model.decode_audio(latents)         
reconstructed = reconstructed.squeeze(0).cpu()  
reconstructed = reconstructed.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

torchaudio.save("reconstructed.wav", reconstructed, sample_rate)
