import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn
from sru import SRU
import numpy as np
from util.utils import calculate_ssnr
import pytorch_lightning as pl
from pesq import pesq

class WaveCRN(nn.Module):
    def __init__(self):
        super(WaveCRN, self).__init__()
        self.net = ConvBSRU(frame_size=96, conv_channels=256, stride=48, num_layers=6, dropout=0.0)

    def forward(self, x):
        return self.net(x)

class ConvBSRU(nn.Module):
    def __init__(self, frame_size, conv_channels, stride=128, num_layers=1, dropout=0.1, rescale=False, bidirectional=True):
        super(ConvBSRU, self).__init__()
        num_directions = 2 if bidirectional else 1
        if stride == frame_size:
            padding = 0
        elif stride == frame_size // 2:
            padding = frame_size // 2
        else:
            print(stride, frame_size)
            raise ValueError(
                'Invalid stride {}. Length of stride must be "frame_size" or "0.5 * "frame_size"'.format(stride))
            
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=conv_channels, 
            kernel_size=frame_size, 
            stride=stride,
            padding=padding,
            bias=False
        )
        self.deconv = nn.ConvTranspose1d(
            in_channels=conv_channels,
            out_channels=1,
            kernel_size=frame_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.outfc = nn.Linear(num_directions * conv_channels, conv_channels, bias=False)
        self.sru = SRU(
            input_size=conv_channels,
            hidden_size=conv_channels,
            num_layers=num_layers,
            dropout=dropout,
            rnn_dropout=0.1,
            layer_norm=True,
            rescale=rescale,
            bidirectional=bidirectional
        )

    def forward(self, x):
        output = self.conv(x) # B,C,D
        output_ = output.permute(2, 0, 1) # D, B, C
        output, _ = self.sru(output_) # D, B, 2C
        output = self.outfc(output) # D, B, C
        #output = output_ * F.sigmoid(output)
        output = output_ * output # D, B, C
        output = output.permute(1, 2, 0) # B, C, D
        output = self.deconv(output)
        #output = self.conv11(output)
        output = torch.tanh(output)

        return output
    
class WaveCRNLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = WaveCRN()
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.model(x)  # Kỳ vọng x: (B, 1, D)

    def training_step(self, batch, batch_idx):
        noisy, clean = batch  # noisy, clean: (B, 1, D)
        denoised = self(noisy)
        loss = self.loss_fn(denoised, clean)
        if batch_idx % 200 == 0:
            self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = self.loss_fn(denoised, clean)

        # Tính PESQ
        clean_np = clean.squeeze(1).cpu().numpy()  # (B, D)
        denoised_np = denoised.squeeze(1).cpu().numpy()  # (B, D)
        pesq_score = np.mean([pesq(16000, c, d, 'wb') for c, d in zip(clean_np, denoised_np)])

    
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log("val_pesq", pesq_score, prog_bar=True, logger=True, sync_dist=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        noisy, clean = batch
        denoised = self(noisy)
        loss = self.loss_fn(denoised, clean)
        clean_np = clean.squeeze(1).cpu().numpy()  # (B, D)
        denoised_np = denoised.squeeze(1).cpu().numpy()  # (B, D)
        pesq_score = np.mean([pesq(16000, c, d, 'wb') for c, d in zip(clean_np, denoised_np)])
        ssnr = np.mean([calculate_ssnr(c, d) for c, d in zip(clean, denoised)])
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("test_pesq", pesq_score, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)
        self.log("test_ssnr", ssnr, prog_bar=True, logger=True, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == "__main__":
    from dataloader.datamodule import AudioDataModule
    data_module = AudioDataModule("clean_audio/", "noisy_audio/", 1)
    data_loader = data_module.train_dataloader()
    noisy, clean = next(iter(data_loader))
    wavecrn = WaveCRN()
    denoised = wavecrn(noisy)
    print(denoised.shape)