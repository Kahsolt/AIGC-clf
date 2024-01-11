#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

from huggingface.aekl import get_sd_vae_ft_ema, get_sd_vae_ft_mse, infer_autoencoder_kl, infer_autoencoder_kl_with_latent_noise
from huggingface.utils import *
from utils import *


@torch.inference_mode()
def stats():
  app = 1
  if app == 0:
    model_func = get_sd_vae_ft_ema
    infer_func = infer_autoencoder_kl
    exp_name = 'aekl_ema'
  elif app == 1:
    model_func = get_sd_vae_ft_mse
    infer_func = infer_autoencoder_kl
    exp_name = 'aekl_mse'
  if app == 2:
    model_func = get_sd_vae_ft_ema
    infer_func = lambda model, img: infer_autoencoder_kl_with_latent_noise(model, img, 1e-3)
    exp_name = 'aekl_ema_noise'

  truth = load_truth()
  model = model_func().to(device)
  loss_fn = lambda x, y: F.l1_loss(x, y).item()

  db_file = OUT_PATH / f'stats_{exp_name}.json'
  db = load_db(db_file)
  fps = get_test_fps()

  for i, fp in enumerate(tqdm(fps)):
    if str(i) in db and type(db[str(i)]) != str: continue
    img = Image.open(fp).convert('RGB')
    x, x_hat = infer_func(model, img)
    err0 = loss_fn(x, x_hat)
    err1 = loss_fn(x, x_hat.clip(0, 1))
    db[i] = (err0, err1)
    print(f'[{i}] truth', truth[i], 'err:', db[i])

  save_db(db, db_file)

  color = []
  err_0 = []
  err_1 = []
  for idx, res in db.items():
    if type(res) == str: continue
    color.append(truth[int(idx)])
    err_0.append(res[0])
    err_1.append(res[1])
  plt.clf()
  plt.scatter(err_0, err_1, c=color, cmap='bwr')
  plt.xlabel('l1_err w/o clip')
  plt.ylabel('l1_err w/ clip')
  plt.suptitle(exp_name)
  plt.savefig(IMG_PATH / f'stats_{exp_name}.png', dpi=800)


if __name__ == '__main__':
  stats()
