from utils import *

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as F
from mindspore import Tensor

ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
