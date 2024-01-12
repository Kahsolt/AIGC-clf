#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

from shutil import copy2

from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule, Trainer

from huggingface.utils import *   # allow overwrite
from huggingface.resnet import get_resnet18_finetuned_ai_art, ResNetForImageClassification, transform_train, transform_valid
from utils import *               # allow overwrite
from predict import predict, get_args

SRC_PATH = HF_PATH / 'artfan123#resnet-18-finetuned-ai-art'
DST_PATH = HF_PATH / 'kahsolt#resnet-18-finetuned-ai-art_ft' ; DST_PATH.mkdir(exist_ok=True)

DTYPE = 'bf16-mixed'
EPOCH = 30
BATCH_SIZE = 32
SPLIT_RATIO = 0.3
FEAT_LR = 1e-4
CLF_LR = 1e-5
MOMENTUM = 0.9


class ImageDataset(Dataset):

  def __init__(self, dp:Path, split:str='train', split_ratio:float=0.7, transform:Callable=None):
    super().__init__()

    fps = get_test_fps(dp)
    truth = load_truth()
    assert len(fps) == len(truth)
    nlen = len(fps)
    cp = round(nlen * split_ratio)
    if split == 'train':
      fps = fps[:cp]
      truth = truth[:cp]
    elif split == 'val':
      fps = fps[cp:]
      truth = truth[cp:]
    elif split == 'all':
      pass

    self.fps = fps
    self.truth = truth
    self.transform = transform
  
  def __len__(self):
    return len(self.fps)

  def __getitem__(self, idx:int):
    y = self.truth[idx]
    img = Image.open(self.fps[idx]).convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img, 1 - y   # swap 0-1


class LitModel(LightningModule):

  def __init__(self, model:ResNetForImageClassification):
    super().__init__()
    self.model = model

  def configure_optimizers(self):
    params = [
      {'params': self.model.resnet.parameters(), 'lr': FEAT_LR},
      {'params': self.model.classifier.parameters(), 'lr': CLF_LR},
    ]
    return SGD(params, momentum=MOMENTUM)

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, y = batch
    logits = self.model(x)
    loss = F.cross_entropy(logits, y)
    if batch_idx % 10 == 0:
      self.log("train_loss", loss)
    return loss
  
  def infer_step(self, batch:Tuple[Tensor]) -> Tuple[Tensor, float]:
    x, y = batch
    logits = self.model(x)
    loss = F.cross_entropy(logits, y)
    pred = torch.argmax(logits, dim=-1)
    acc = torch.sum(y == pred).item() / len(y)
    return loss, acc

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int):
    loss, acc = self.infer_step(batch)
    if batch_idx % 10 == 0:
      self.log_dict({'val_loss': loss.item(), 'val_acc': acc})

  def test_step(self, batch:Tuple[Tensor], batch_idx:int):
    loss, acc = self.infer_step(batch)
    if batch_idx % 10 == 0:
      self.log_dict({'test_loss': loss.item(), 'test_acc': acc})


def get_dataloaders(batch_size:int, split_ratio:float, transform_train:Callable, transform_valid:Callable) -> Tuple[DataLoader, DataLoader, DataLoader]:
  trainset = ImageDataset(DATA_PATH, 'train', split_ratio, transform=transform_train)
  trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, persistent_workers=True, num_workers=2)
  validset = ImageDataset(DATA_PATH, 'valid', split_ratio, transform=transform_valid)
  validloader = DataLoader(validset, batch_size=batch_size, pin_memory=True, persistent_workers=True, num_workers=2)
  dataset = ImageDataset(DATA_PATH, 'all', split_ratio, transform=transform_valid)
  dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, persistent_workers=True, num_workers=2)
  return trainloader, validloader, dataloader


def train():
  model = get_resnet18_finetuned_ai_art()

  lit = LitModel(model)
  trainloader, validloader, dataloader = get_dataloaders(BATCH_SIZE, SPLIT_RATIO, transform_train, transform_valid)
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
  trainer.test(lit, dataloader, 'best')

  lit = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, model=model)
  np.savez(DST_PATH / 'model.npz', **{k: v.cpu().numpy() for k, v in lit.model.state_dict().items()})
  copy2(SRC_PATH / 'model.json', DST_PATH / 'model.json')

  predict(get_args(), app=1)


if __name__ == '__main__':
  train()
