from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning import seed_everything

torch.set_float32_matmul_precision('medium')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(42)
