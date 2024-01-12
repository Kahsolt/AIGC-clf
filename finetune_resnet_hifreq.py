#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/12

from finetune_resnet import *
from torch.optim.lr_scheduler import OneCycleLR
from huggingface.resnet import transform_train_highfreq, transform_valid_highfreq

SRC_PATH = HF_PATH / 'artfan123#resnet-18-finetuned-ai-art'
DST_PATH = HF_PATH / 'kahsolt#resnet-18-finetuned-ai-art_hf' ; DST_PATH.mkdir(exist_ok=True)

DTYPE = 'bf16-mixed'
EPOCH = 150
BATCH_SIZE = 32
SPLIT_RATIO = 0.7
INPUT_LR = 1e-4
FEAT_LR = 1e-5
CLF_LR = 1e-4
MAX_LR_LIST = [
  1e-3, 
  1e-4, 
  1e-3,
]
MOMENTUM = 0.9


class ImageHiFreqDataset(ImageDataset):

  def __getitem__(self, idx:int):
    y = self.truth[idx]
    img = load_img(self.fps[idx])
    im_hi = to_hifreq(img)
    if self.transform:
      im_hi = im_to_tensor(im_hi)
      im_hi = self.transform(im_hi)
    return im_hi, 1 - y   # swap 0-1


class LitModelHF(LitModel):

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
    sched = OneCycleLR(optim, max_lr=MAX_LR_LIST, total_steps=total_steps, final_div_factor=1e2, verbose=True)
    return {
      'optimizer': optim,
      'lr_scheduler': sched,
    }


if __name__ == '__main__':
  model = get_resnet18_finetuned_ai_art()

  ckpt_path = None
  #ckpt_path = BASE_PATH / 'lightning_logs/version_19/checkpoints/epoch=149-step=3300.ckpt'

  trainloader, validloader, dataloader = get_dataloaders(BATCH_SIZE, SPLIT_RATIO, transform_train_highfreq, transform_valid_highfreq, ImageHiFreqDataset)
  total_steps = EPOCH * len(trainloader)
  lit = LitModelHF(model, total_steps)
  trainer = Trainer(
    max_epochs=EPOCH,
    precision=DTYPE,
    benchmark=True,
    enable_checkpointing=True,
    log_every_n_steps=10,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
  )
  trainer.fit(lit, trainloader, validloader, ckpt_path=ckpt_path)
  lit = LitModelHF.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model)
  trainer.test(lit, dataloader, 'best')

  np.savez(DST_PATH / 'model.npz', **{k: v.cpu().numpy() for k, v in lit.model.state_dict().items()})
  copy2(SRC_PATH / 'model.json', DST_PATH / 'model.json')

  predict(get_args(), app=3)
