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

IS_FOR_ENSEMBLE = True
DTYPE = 'bf16-mixed'
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
    elif split in ['all', '']:
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


def get_dataloaders(batch_size:int, split_ratio:float, transform:Callable) -> Tuple[DataLoader, DataLoader]:
  kwargs = {
    'batch_size': batch_size, 
    'pin_memory': True, 
    'num_workers': 2,
    'persistent_workers': True,
  }
  trainset = ImageDataset(DATA_PATH, 'train', split_ratio, transform=transform)
  trainloader = DataLoader(trainset, **kwargs)
  validset = ImageDataset(DATA_PATH, 'valid', split_ratio, transform=transform)
  validloader = DataLoader(validset, **kwargs)
  print('trainset:', len(trainset), 'trainloader:', len(trainloader))
  print('validset:', len(validset),'validloader:', len(validloader))
  return trainloader, validloader


class ImageDatasetForEnsemble(Dataset):

  def __init__(self, dp:Path, idx:int, split:str='train', transform:Callable=None):
    super().__init__()

    assert idx in [0, 1, 2]
    TRAIN_RATIO = 0.3
    OVERLAP_RATIO = 0.1
    VALID_RATIO = 0.4

    all_fps = get_test_fps(dp)
    all_truth = load_truth()
    assert len(all_fps) == len(all_truth)
    nlen = len(all_fps)

    cp = round(nlen * VALID_RATIO)
    # first 60% as trainset, but assign 30% to each model with 10% pairwise overlap
    train_fps, train_truth = all_fps[:cp], all_truth[:cp]
    # last 40% as common validtest
    valid_fps, valid_truth = all_fps[:-cp], all_truth[:-cp]
    if split in ['val', 'valid']:
      fps = valid_fps
      truth = valid_truth
    elif split == 'train':
      cp_start = round(nlen * max(0.0, idx * TRAIN_RATIO - OVERLAP_RATIO))
      cp_end = cp_start + round(nlen * TRAIN_RATIO)
      fps = (train_fps + train_fps)[cp_start:cp_end]
      truth = (train_truth + train_truth)[cp_start:cp_end]
    else: raise ValueError(split)

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


def get_dataloaders_for_ensemble(batch_size:int, idx:int, transform:Callable) -> Tuple[DataLoader, DataLoader]:
  kwargs = {
    'batch_size': batch_size, 
    'pin_memory': True, 
    'num_workers': 2,
    'persistent_workers': True,
  }
  trainset = ImageDatasetForEnsemble(DATA_PATH, idx, 'train', transform=transform)
  trainloader = DataLoader(trainset, **kwargs)
  validset = ImageDatasetForEnsemble(DATA_PATH, idx, 'valid', transform=transform)
  validloader = DataLoader(validset, **kwargs)
  print('trainset:', len(trainset), 'trainloader:', len(trainloader))
  print('validset:', len(validset),'validloader:', len(validloader))
  return trainloader, validloader


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
