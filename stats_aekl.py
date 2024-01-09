#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

from utils import *

from models.aekl import get_sd_vae_ft_ema, get_sd_vae_ft_mse, infer_autoencoder_kl


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

  db_file = OUT_PATH / f'stats_{model_name}.json'
  db = load_db(db_file)
  fps = get_test_fps()

  gc.enable()
  try:
    for i, fp in enumerate(tqdm(fps)):
      if str(i) in db and type(db[str(i)]) != str: continue
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
      save_db(db, db_file)
  except KeyboardInterrupt:
    pass
  gc.disable()

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
  plt.suptitle(model_name)
  plt.savefig(IMG_PATH / f'stats_{model_name}.png', dpi=800)


if __name__ == '__main__':
  stats()
