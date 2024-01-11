#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore.train.callback import LossMonitor
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

from utils import *

# https://zhuanlan.zhihu.com/p/457516369

EPOCHS = 10

class MyNet(nn.Cell):
    
  def __init__(self):
    super().__init__()
    self.fc = nn.Dense(2, 2)

  def construct(self, x:Tensor) -> Tensor:
    return self.fc(x)


preds = list(load_db(RESULT_FILE).values())
X = np.stack(preds, axis=0)
Y = np.asarray(load_truth())
dataset = GeneratorDataset(zip(X, Y), column_names=['data', 'label'])
dataloader = dataset.batch(32)
net = MyNet().set_train()
loss_fn = nn.CrossEntropyLoss()
optim = nn.Momentum(net.trainable_params(), learning_rate=0.001, momentum=0.7, weight_decay=1e-5)
model = ms.Model(net, loss_fn, optim, metrics={'accuracy'})
model.train(EPOCHS, dataloader, callbacks=LossMonitor())

tot, ok = 0, 0
for X, Y in dataloader.create_tuple_iterator():
  output = net(X)
  pred = output.argmax(-1)
  ok += (Y == pred).sum().item()
  tot += len(Y)

print(f'Accuracy: {ok / tot:.5%}')
