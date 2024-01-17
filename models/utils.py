#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/11 

import math
import json
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from pathlib import Path
from functools import partial
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as T
from lightning import seed_everything
import numpy as np
from numpy import ndarray

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16
seed_everything(42)

npimg = ndarray

HF_PATH = Path(__file__).parent.parent / 'huggingface'

ACT2FN = {
  "gelu": nn.GELU(),
  "linear": nn.Identity(),
  "mish": nn.Mish(),
  "relu": nn.ReLU(),
  "relu6": nn.ReLU6(),
  "sigmoid": nn.Sigmoid(),
  "silu": nn.SiLU(),
  "swish": nn.SiLU(),
  "tanh": nn.Tanh(),
}


def load_img(fp:Path) -> PILImage:
  return Image.open(fp).convert('RGB')

def pil_to_npimg(img:PILImage) -> npimg:
  return np.asarray(img, dtype=np.uint8)

def npimg_to_pil(im:npimg) -> PILImage:
  assert im.dtype in [np.uint8, np.float32]
  if im.dtype == np.float32:
    assert 0.0 <= im.min() and im.max() <= 1.0
  return Image.fromarray(im)

def im_to_tensor(im:ndarray) -> Tensor:
  return torch.from_numpy(im).permute([2, 0, 1])

def tensor_to_im(X:Tensor) -> ndarray:
  return X.permute([1, 2, 0]).cpu().numpy()

def minmax_norm(dx:ndarray, vmin:int=None, vmax:int=None) -> npimg:
  vmin = vmin or dx.min()
  vmax = vmax or dx.max()
  out = (dx - vmin) / (vmax - vmin)
  return (out * 255).astype(np.uint8)

def npimg_diff(x:npimg, y:npimg) -> ndarray:
  return x.astype(np.int16) - y.astype(np.int16)

def to_hifreq(img:PILImage) -> ndarray:
  img_lo = img.filter(ImageFilter.GaussianBlur(3))
  im_lo = pil_to_npimg(img_lo)
  im = pil_to_npimg(img)
  im_hi = npimg_diff(im, im_lo)   # int16
  return (im_hi / 255.0).astype(np.float32)


def infer_clf(model:nn.Module, X:Tensor, debug:bool=False) -> Union[int, Tuple[float, float], Tuple[int, int], Tuple[int]]:
  logits = model(X.unsqueeze(0)).squeeze(0)
  probs = F.softmax(logits, dim=-1)
  pred = torch.argmax(probs).item()
  return (logits.cpu().numpy().tolist(), probs.cpu().numpy().tolist(), pred) if debug else pred

def infer_clf_batch(model:nn.Module, X:Tensor) -> List[int]:
  assert len(X.shape) == 4, f'>> need BCWH format, but got {X.shape}'
  logits = model(X)
  preds = torch.argmax(logits, dim=-1)
  return preds.cpu().numpy().tolist()
