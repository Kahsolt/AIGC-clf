#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/11 

import math
import json
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

HF_PATH = Path(__file__).parent

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


def im_to_tensor(im:ndarray) -> Tensor:
  return torch.from_numpy(im).permute([2, 0, 1])

def tensor_to_im(X:Tensor) -> ndarray:
  return X.permute([1, 2, 0]).cpu().numpy()
