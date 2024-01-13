#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/04 

from models.aekl import AutoencoderKL, DiagonalGaussianDistribution
from models.utils import *

opt_f = 8
PATCH_SIZE = 256
LATENT_SIZE = PATCH_SIZE // opt_f

transform = T.Compose([
  T.RandomResizedCrop((PATCH_SIZE, PATCH_SIZE), scale=(0.8, 1.0)),
  T.RandomHorizontalFlip(),
  T.RandomVerticalFlip(),
  T.ToTensor(),
])


class AutoencoderKLClassifier(nn.Module):

  def __init__(self, model: AutoencoderKL):
    super().__init__()

    self.encoder = model.encoder
    self.quant_conv = model.quant_conv
    self.clf = nn.Sequential(
      nn.Linear(4 * LATENT_SIZE * LATENT_SIZE, 256),
      nn.SiLU(inplace=True),
      nn.Linear(256, 128),
      nn.SiLU(inplace=True),
      nn.Linear(128, 2),
    )

  def encode(self, x:Tensor) -> Tensor:
    h = self.encoder(x)
    moments = self.quant_conv(h)
    z = DiagonalGaussianDistribution(moments).mode()
    return z

  def forward(self, x:Tensor) -> Tensor:
    z = self.encode(x)   # [B, C=4, H=32, W=32]
    z = z.flatten(start_dim=1)
    o = self.clf(z)
    return o


def get_app(app_name:str, aekl:AutoencoderKL) -> AutoencoderKLClassifier:
  APP_PATH = HF_PATH / app_name
  WEIGHT_FILE = APP_PATH / 'model.npz'

  model = AutoencoderKLClassifier(aekl)
  model = model.eval()
  weights = np.load(WEIGHT_FILE)
  state_dict = {k: torch.from_numpy(v) for k, v in weights.items()}
  model.load_state_dict(state_dict)
  return model


def get_app_aekl_clf():
  from models.aekl import get_app_vae_ema
  aekl = get_app_vae_ema()
  return get_app('kahsolt#aekl-clf', aekl)


if __name__ == '__main__':
  model = get_app_aekl_clf()
  print(model)
  X = torch.zeros([1, 3, PATCH_SIZE, PATCH_SIZE])
  logits = model(X)
  print(logits)
