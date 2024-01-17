#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/12

from train_utils import *
from models.resnet import get_app_resnet, ResNetForImageClassification, transform_highfreq
from predict import predict, get_args

APP_NAME = 'resnet-hf'
SRC_PATH = HF_PATH / 'artfan123#resnet-18-finetuned-ai-art'
DST_PATH = HF_PATH / f'kahsolt#{APP_NAME}' ; DST_PATH.mkdir(exist_ok=True)

EPOCH = 100
BATCH_SIZE = 32
SPLIT_RATIO = 0.3
INPUT_LR = 1e-4
FEAT_LR = 1e-4
CLF_LR = 1e-4
MAX_LR_LIST = [
  1e-3,
  1e-3,
  1e-3,
]


class MyLitModel(LitModel):

  def __init__(self, model:ResNetForImageClassification, total_steps:int=-1):
    super().__init__(model)

    self.model = model
    self.total_steps = total_steps

  def make_trainable_params(self) -> List[Dict[str, Any]]:
    return [
      {'params': self.model.resnet.embedder.parameters(), 'lr': INPUT_LR},
      {'params': self.model.resnet.encoder .parameters(), 'lr': FEAT_LR},
      {'params': self.model.classifier     .parameters(), 'lr': CLF_LR},
    ]

  def configure_optimizers(self):
    params = self.make_trainable_params()
    optim = AdamW(params)
    sched = OneCycleLR(optim, max_lr=MAX_LR_LIST, total_steps=self.total_steps, final_div_factor=1e2, verbose=True)
    return {
      'optimizer': optim,
      'lr_scheduler': sched,
    }


if __name__ == '__main__':
  trainloader, validloader = get_dataloaders_for_ensemble(BATCH_SIZE, 1, transform_highfreq) if IS_FOR_ENSEMBLE else get_dataloaders(BATCH_SIZE, SPLIT_RATIO, transform_highfreq)
  total_steps = EPOCH * len(trainloader)
  model = get_app_resnet()
  lit = MyLitModel(model, total_steps)
  trainer = Trainer(
    max_epochs=EPOCH,
    precision=DTYPE,
    benchmark=True,
    enable_checkpointing=True,
    log_every_n_steps=10,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
  )
  trainer.fit(lit, trainloader, validloader)
  lit = MyLitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model)
  save_npz_weights(lit.model, SRC_PATH, DST_PATH)

  args = get_args()
  predict(args, app_name=APP_NAME)
  args.votes = 7
  predict(args, app_name=APP_NAME)
