import os
import torch
import torchaudio
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from accelerate import Accelerator
from utils import pad_wav
from stable_audio_tools.models.pretrained import get_pretrained_model

def read_wav_file(filename, duration_sec):
    info = torchaudio.info(filename)
    sample_rate = info.sample_rate
    num_frames = int(sample_rate * duration_sec)

    waveform, sr = torchaudio.load(filename, num_frames=num_frames)

    # Resample
    if sr != 44100:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
        waveform = resampler(waveform)

    # Convert mono to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Pad each channel
    target_length = int(44100 * duration_sec)
    padded_left = pad_wav(waveform[0], target_length)
    padded_right = pad_wav(waveform[1], target_length)

    return torch.stack([padded_left, padded_right])


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # ===== 修改以下路径以适配你的数据 =====
    input_dir = "test_vae_data" # "/cfs-r3ufsqcb/for_share/datasets/music_restoration_data/label_data"
    output_dir = "test_vae_data_latents" # "/cfs-r3ufsqcb/for_share/datasets/music_restoration_data/label_latents_stable_audio_3"
    # ========================================

    duration_sec = 10
    batch_size = 56
    
    print("load model")
    # Load VAE and prepare for multi-GPU
    # vae = AutoencoderOobleck.from_pretrained(
    #     "stabilityai/stable-audio-open-1.0", subfolder="vae"
    # )
    vae, model_config = get_pretrained_model("stabilityai/SAME-L")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    vae.eval()
    vae.requires_grad_(False)
    vae = accelerator.prepare(vae)
    
    # Recursively find all .wav files in input_dir using os.walk (faster than glob)
    print("load wav")
    wav_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    wav_files = sorted(wav_files)

    # Partition file list by rank
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    wav_files = wav_files[rank::world_size]  # Distribute files across processes

    batch_waveforms = []
    batch_filenames = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print("prepare infering")
    for wav_file in tqdm(wav_files, desc=f"Rank {rank} Encoding", disable=not accelerator.is_main_process):
        try:
            waveform = read_wav_file(wav_file, duration_sec)
            batch_waveforms.append(waveform)
            batch_filenames.append(os.path.basename(wav_file))

            if len(batch_waveforms) == batch_size or wav_file == wav_files[-1]:
                batch_tensor = torch.stack(batch_waveforms).to(device)

                with torch.no_grad():
                    # latents = vae.encode(batch_tensor).latent_dist.mode()
                    latents = vae.encode_audio(batch_tensor)
                    latents = latents.transpose(1, 2)  # [B, T, C]
                    print("shape: ", latents.shape)
                for fname, latent in zip(batch_filenames, latents.cpu()):
                    outpath = os.path.join(output_dir, fname.replace(".wav", ".pt"))
                    torch.save(latent, outpath)

                batch_waveforms.clear()
                batch_filenames.clear()

        except Exception as e:
            print(f"Error processing {wav_file} on rank {rank}: {e}")


if __name__ == "__main__":
    main()

