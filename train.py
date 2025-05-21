from model import WaveCRNLightning
from dataloader.datamodule import AudioDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

# Hàm chính để huấn luyện
def main():
    # Đường dẫn dữ liệu
    clean_dir = "/path/to/clean_audios/"
    noisy_dir = "/path/to/noisy_audios/"
    checkpoint_dir = "/path/to/checkpoints/"

    # Khởi tạo WandB
    wandb.login()  # Đăng nhập WandB (cần API key trên Kaggle)
    logger = WandbLogger(project="wavecrn_denoising", name="wavecrn_run")

    # Khởi tạo DataModule
    data_module = AudioDataModule(clean_dir, noisy_dir, batch_size=16)

    # Khởi tạo mô hình
    model = WaveCRNLightning()

    # Cấu hình callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="wavecrn-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Cấu hình trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=200,
        val_check_interval=2000
    )

    # Huấn luyện mô hình
    trainer.fit(model, datamodule=data_module)

    # Đánh giá trên tập test
    trainer.test(model, datamodule=data_module)

    # Kết thúc WandB run
    wandb.finish()

if __name__ == "__main__":
    main()