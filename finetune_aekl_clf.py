#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/11 

from finetune_resnet import *   # import firstly, allow overwrite
from huggingface.aekl_clf import get_sd_vae_ft_ema, AutoencoderKLClassifier, transform_train, transform_valid

SRC_PATH = HF_PATH / 'stabilityai#sd-vae-ft-ema'
DST_PATH = HF_PATH / 'kahsolt#sd-vae-ft-ema_clf' ; DST_PATH.mkdir(exist_ok=True)

EPOCH = 100
BATCH_SIZE = 4
FEAT_LR = 1e-4
CLF_LR = 1e-3
MOMENTUM = 0.9


class LitModelAEKL(LitModel):

  def __init__(self, model:AutoencoderKLClassifier):
    super().__init__(model)
    self.model = model

  def configure_optimizers(self):
    params = [
      {'params': self.model.model.parameters(), 'lr': FEAT_LR},
      {'params': self.model.mlp.parameters(), 'lr': CLF_LR},
    ]
    return SGD(params, momentum=MOMENTUM)


def train():
  aekl = get_sd_vae_ft_ema()
  model = AutoencoderKLClassifier(aekl)

  lit = LitModelAEKL(model)
  trainloader, validloader, dataloader = get_dataloaders(BATCH_SIZE, transform_train, transform_valid)
  trainer = Trainer(
    max_epochs=EPOCH,
    precision='bf16-mixed',
    benchmark=True,
    enable_checkpointing=True,
    log_every_n_steps=10,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
  )
  trainer.fit(lit, trainloader, validloader)
  trainer.test(lit, dataloader, 'best')

  lit = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model)
  np.savez(DST_PATH / 'model.npz', **{k: v.cpu().numpy() for k, v in lit.model.state_dict().items()})
  copy2(SRC_PATH / 'model.json', DST_PATH / 'model.json')


if __name__ == '__main__':
  train()
