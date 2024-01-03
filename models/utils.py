#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

import json
from pathlib import Path
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


def load_npz_as_param_dict(fp:Path) -> Dict[str, Parameter]:
  kv: Dict[str, ndarray] = np.load(fp, allow_pickle=True)
  return {k: Parameter(v) for k, v in kv.items()}
