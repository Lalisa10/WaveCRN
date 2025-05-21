import glob
from torch.utils.data import Dataset
import os
import torch
import torchaudio
import numpy as np

class CustomAudioDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000):
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
        self.sample_rate = sample_rate
        assert len(self.clean_files) == len(self.noisy_files), "Số file clean và noisy không khớp"

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Tải file âm thanh
        clean_waveform, sr = torchaudio.load(self.clean_files[idx])
        noisy_waveform, sr = torchaudio.load(self.noisy_files[idx])

        # Đảm bảo sample rate đúng
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            clean_waveform = resampler(clean_waveform)
            noisy_waveform = resampler(noisy_waveform)

        # Đảm bảo mono channel
        if clean_waveform.shape[0] > 1:
            clean_waveform = clean_waveform.mean(dim=0, keepdim=True)
            noisy_waveform = noisy_waveform.mean(dim=0, keepdim=True)

        # Cắt hoặc đệm để đảm bảo độ dài chia het 48
        assert noisy_waveform.size(-1) == clean_waveform.size(-1)
        pad_length = 48 - noisy_waveform.size(-1) % 48
        noisy_padded = torch.nn.functional.pad(noisy_waveform, (0, pad_length))
        clean_padded = torch.nn.functional.pad(noisy_waveform, (0, pad_length))
        print(noisy_padded.shape)
        return noisy_padded, clean_padded
    
if __name__ == "__main__":
    dataset = CustomAudioDataset('clean_audio/', 'noisy_audio/')
    print(dataset[0])
