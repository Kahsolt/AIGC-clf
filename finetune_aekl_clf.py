#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/11 

from huggingface.aekl_clf import get_sd_vae_ft_ema, AutoencoderKLClassifier, LATENT_SIZE
from finetune_resnet import *

DST_PATH = HF_PATH / 'stabilityai#sd-vae-ft-ema_finetune' ; DST_PATH.mkdir(exist_ok=True)

EPOCH = 100
BATCH_SIZE = 32
LR = 1e-3
MOMENTUM = 0.8

transform_train = T.Compose([
  T.RandomResizedCrop((PATCH_SIZE, PATCH_SIZE)),
  T.RandomHorizontalFlip(),
  T.ToTensor(),
])

transform_valid = T.Compose([
  T.RandomResizedCrop((PATCH_SIZE, PATCH_SIZE)),
  T.ToTensor(),
])


class LitModelAEKL(LitModel):

  def configure_optimizers(self):
    return SGD(self.parameters(), lr=LR, momentum=MOMENTUM)


def train():
  aekl = get_sd_vae_ft_ema()
  model = AutoencoderKLClassifier(aekl, LATENT_SIZE)

  lit = LitModelAEKL(model)
  lit.data = LightningDataModule.from_datasets(
    train_dataset=ImageDataset(DATA_PATH, 'train', transform=transform_train),
    val_dataset=ImageDataset(DATA_PATH, 'val', transform=transform_valid),
    num_workers=2,
  )
  trainer = Trainer(
    max_epochs=EPOCH,
    precision='bf16-mixed',
    benchmark=True,
    enable_checkpointing=False,
    log_every_n_steps=10,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
  )
  trainer.fit(lit, datamodule=lit.data)

  fp = DST_PATH / 'model.npz'
  print(f'>> save to {fp}')
  np.savez(fp, **{k: v.cpu().numpy() for k, v in lit.model.state_dict().items()})
  fp = DST_PATH / 'model.json'
  print(f'>> copy to {fp}')
  copy2(CONFIG_FILE, fp)


if __name__ == '__main__':
  train()
