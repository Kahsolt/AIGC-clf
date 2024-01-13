#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/13 

from shutil import copy2

from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader
from lightning import LightningModule, Trainer

from models.utils import *   # init torch
from utils import *

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
    img = load_img(self.fps[idx])
    if self.transform:
      img = self.transform(img)
    return img, 1 - y   # swap 0-1


class ImageHiFreqDataset(ImageDataset):

  def __getitem__(self, idx:int):
    y = self.truth[idx]
    img = load_img(self.fps[idx])
    im_hi = to_hifreq(img)
    if self.transform:
      im_hi = im_to_tensor(im_hi)
      im_hi = self.transform(im_hi)
    return im_hi, 1 - y   # swap 0-1


def get_dataloaders(batch_size:int, split_ratio:float, transform_train:Callable, transform_valid:Callable, dataset_cls=ImageDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
  kwargs = {
    'batch_size': batch_size, 
    'pin_memory': True, 
    'num_workers': 2,
    'persistent_workers': True,
  }
  trainset = dataset_cls(DATA_PATH, 'train', split_ratio, transform=transform_train)
  trainloader = DataLoader(trainset, **kwargs)
  validset = dataset_cls(DATA_PATH, 'valid', split_ratio, transform=transform_valid)
  validloader = DataLoader(validset, **kwargs)
  dataset = dataset_cls(DATA_PATH, 'all', split_ratio, transform=transform_valid)
  dataloader = DataLoader(dataset, **kwargs)
  return trainloader, validloader, dataloader


class LitModel(LightningModule):

  def __init__(self, model:nn.Module):
    super().__init__()

    self.model = model

  def make_trainable_params(self):
    raise NotImplementedError

  def configure_optimizers(self):
    return SGD(self.make_trainable_params(), momentum=MOMENTUM)

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


def save_npz_weights(model:LitModel, src:Path, dst:Path):
  np.savez(dst / 'model.npz', **{k: v.cpu().numpy() for k, v in model.state_dict().items()})
  copy2(src / 'model.json', dst / 'model.json')
