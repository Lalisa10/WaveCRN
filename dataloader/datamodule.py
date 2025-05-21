import pytorch_lightning as pl
from .dataset import CustomAudioDataset
from torch.utils.data import DataLoader, random_split

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, clean_dir, noisy_dir, batch_size=16, sample_rate=16000):
        super().__init__()
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.num_workers = 4
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 2

    def setup(self, stage=None):
        # Tạo dataset
        full_dataset = CustomAudioDataset(self.clean_dir, self.noisy_dir, self.sample_rate, self.segment_length)
        
        # Chia dataset thành train/val/test
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )