#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

from shutil import copy2

from torch.optim import SGD
from torch.utils.data import Dataset
import torchvision.transforms as T
from lightning import LightningModule, LightningDataModule, Trainer

from huggingface.resnet import ResNetConfig, ResNetForImageClassification
from huggingface.utils import *
from utils import *

SRC_PATH = HF_PATH / 'artfan123#resnet-18-finetuned-ai-art'
DST_PATH = HF_PATH / 'artfan123#resnet-18-finetuned-ai-art_finetune' ; DST_PATH.mkdir(exist_ok=True)
CONFIG_FILE = SRC_PATH / 'model.json'
WEIGHT_FILE = SRC_PATH / 'model.npz'

EPOCH = 100
BATCH_SIZE = 32
LR = 1e-5
MOMENTUM = 0.8

transform_train = T.Compose([
  T.RandomResizedCrop((224, 224)),
  T.RandomHorizontalFlip(),
  T.ToTensor(),
  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_valid = T.Compose([
  T.ToTensor(),
  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):

  def __init__(self, dp:Path, split:str='train', transform:Callable=None):
    super().__init__()

    fps = get_test_fps(dp)
    truth = load_truth()
    assert len(fps) == len(truth)
    nlen = len(fps)
    # train:val = 7:3
    cp = round(nlen * 0.7)
    if split == 'train':
      fps = fps[:cp]
      truth = truth[:cp]
    elif split == 'val':
      fps = fps[cp:]
      truth = truth[cp:]

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

  def __init__(self, model:nn.Module) -> None:
    super().__init__()

    self.model = model
    self.data = LightningDataModule.from_datasets(
      train_dataset=ImageDataset(DATA_PATH, 'train', transform=transform_train),
      val_dataset=ImageDataset(DATA_PATH, 'val', transform=transform_valid),
    )

  def configure_optimizers(self):
    return SGD(self.parameters(), lr=LR, momentum=MOMENTUM)

  def training_step(self, batch:Tuple[Tensor], batch_idx:int) -> Tensor:
    x, y = batch
    logits = self.model(x)
    loss = F.cross_entropy(logits, y)
    if batch_idx % 10 == 0:
      self.log("train_loss", loss)
    return loss

  def validation_step(self, batch:Tuple[Tensor], batch_idx:int):
    x, y = batch
    logits = self.model(x)
    loss = F.cross_entropy(logits, y)
    pred = torch.argmax(logits, dim=-1)
    acc = torch.sum(y == pred).item() / len(y)
    if batch_idx % 10 == 0:
      self.log_dict({'val_loss': loss.item(), 'val_acc': acc})


def train():
  with open(CONFIG_FILE) as fh:
    cfg = json.load(fh)
  config = ResNetConfig(**cfg)
  model = ResNetForImageClassification(config)
  model = model.eval()
  weights = np.load(WEIGHT_FILE)
  state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}
  model.load_state_dict(state_dict)

  lit = LitModel(model)
  trainer = Trainer(
    max_epochs=EPOCH,
    precision='32',
    benchmark=True,
    enable_checkpointing=False,
    log_every_n_steps=10,
    val_check_interval=1.0,
    check_val_every_n_epoch=1,
  )
  trainer.fit(lit, datamodule=lit.data)

  fp = DST_PATH / 'model.npz'
  print(f'>> save to {fp}')
  np.savez(fp, **{k: v.cpu().numpy() for k, v in lit.model.state_dict().items()})
  fp = DST_PATH / 'model.json'
  print(f'>> copy to {fp}')
  copy2(CONFIG_FILE, fp)


if __name__ == '__main__':
  train()
