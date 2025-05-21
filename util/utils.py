import torch

# Định nghĩa hàm tính SSNR
def calculate_ssnr(clean, denoised, frame_length=400, hop_length=200):
    # Đảm bảo clean và denoised có shape (1, D)
    clean = clean.squeeze(0) if clean.dim() == 3 else clean
    denoised = denoised.squeeze(0) if denoised.dim() == 3 else denoised
    
    # Chia tín hiệu thành các frame
    frames_clean = torch.nn.functional.unfold(
        clean.unsqueeze(0).unsqueeze(-1),
        kernel_size=(frame_length, 1),
        stride=(hop_length, 1)
    ).squeeze(0).T
    frames_denoised = torch.nn.functional.unfold(
        denoised.unsqueeze(0).unsqueeze(-1),
        kernel_size=(frame_length, 1),
        stride=(hop_length, 1)
    ).squeeze(0).T

    # Tính SNR cho mỗi frame
    noise = frames_clean - frames_denoised
    signal_power = torch.mean(frames_clean ** 2, dim=1)
    noise_power = torch.mean(noise ** 2, dim=1)
    snr_per_frame = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    
    # Loại bỏ các frame không hợp lệ và tính trung bình
    valid_snrs = snr_per_frame[torch.isfinite(snr_per_frame)]
    return torch.mean(valid_snrs).item() if len(valid_snrs) > 0 else 0.0