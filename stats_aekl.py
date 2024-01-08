#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from traceback import format_exc
import gc

import mindspore as ms
import mindspore.ops as F
import matplotlib.pyplot as plt
ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)

from models.aekl import get_sd_vae_ft_ema, get_sd_vae_ft_mse, infer_autoencoder_kl
from predict import load_truth

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'test'


def stats():
  app = 0
  if app == 0:
    model_func = get_sd_vae_ft_ema
    infer_func = infer_autoencoder_kl
    model_name = 'aekl_ema'
  elif app == 1:
    model_func = get_sd_vae_ft_mse
    infer_func = infer_autoencoder_kl
    model_name = 'aekl_mse'

  truth = load_truth()
  model = model_func()
  loss_fn = lambda x, y: F.l1_loss(x, y).item()

  DB_FILE = BASE_PATH / 'output' / f'stats_{model_name}.json'
  if DB_FILE.exists():
    with open(DB_FILE, 'r', encoding='utf-8') as fh:
      db = json.load(fh)
  else:
    db = {}

  cmp_fp = lambda fp: int(Path(fp).stem)
  fps = sorted(Path(DATA_PATH).iterdir(), key=cmp_fp)

  gc.enable()
  try:
    for i, fp in enumerate(tqdm(fps)):
      if str(i) in db: continue
      img = Image.open(fp).convert('RGB')
      try:
        x, x_hat = infer_func(model, img)
        err0 = loss_fn(x, x_hat)
        err1 = loss_fn(x, x_hat.clip(0, 1))
        db[i] = (err0, err1)
        print(f'[{i}] truth', truth[i], 'err:', db[i])
      except:
        e = format_exc()
        print(e)
        db[i] = e
        gc.collect()
  except KeyboardInterrupt:
    pass
  gc.disable()

  with open(DB_FILE, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False)

  color = []
  err_0 = []
  err_1 = []
  for idx, res in db.items():
    if type(res) == str: continue
    color.append(truth[int(idx)])
    err_0.append(res[0])
    err_1.append(res[1])
  plt.clf()
  plt.scatter(err_0, err_1, c=color)
  plt.xlabel('l1_err w/o clip')
  plt.ylabel('l1_err w/ clip')
  plt.suptitle(model_name)
  plt.show()


if __name__ == '__main__':
  stats()
