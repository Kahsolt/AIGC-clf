#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

import math
import json
from pathlib import Path
from functools import partial
from typing import *

import numpy as np
from numpy import ndarray
from PIL.Image import Image as PILImage

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Tensor, Parameter
import mindspore.dataset.transforms as T
import mindspore.dataset.vision.transforms as VT

ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
ms.set_seed(42)

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

def load_npz_as_param_dict(fp:Path) -> Dict[str, Parameter]:
  kv: Dict[str, ndarray] = np.load(fp, allow_pickle=True)
  return {k: Parameter(v) for k, v in kv.items()}
