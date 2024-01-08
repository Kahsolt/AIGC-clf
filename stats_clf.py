#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import gc

import mindspore as ms
import matplotlib.pyplot as plt
ms.set_context(device_target='CPU', mode=ms.PYNATIVE_MODE)

from models.resnet import get_resnet18_finetuned_ai_art, infer_resnet
from models.swin import get_AI_image_detector, get_xl_detector, infer_swin
from predict import load_truth

BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / 'test'


def stats():
  app = 0
  if app == 0:
    model_func = get_resnet18_finetuned_ai_art
    infer_func = infer_resnet
    model_name = 'resnet'
  elif app == 1:
    model_func = get_AI_image_detector
    infer_func = infer_swin
    model_name = 'swin_sd'
  elif app == 2:
    model_func = get_xl_detector
    infer_func = infer_swin
    model_name = 'swin_sdx'

  truth = load_truth()
  model = model_func()

  DB_FILE = BASE_PATH / 'output' / f'stats_{model_name}.json'
  if DB_FILE.exists():
    with open(DB_FILE, 'r', encoding='utf-8') as fh:
      db = json.load(fh)
  else:
    db = {}

  cmp_fp = lambda fp: int(Path(fp).stem)
  fps = sorted(Path(DATA_PATH).iterdir(), key=cmp_fp)

  gc.enable()
  for i, fp in enumerate(tqdm(fps)):
    if str(i) in db: continue
    img = Image.open(fp).convert('RGB')
    logits, probs, pred = infer_func(model, img, debug=True)
    db[i] = {
      'logits': logits,
      'probs': probs,
      'pred': pred,
      'ok': truth[i] == 1 - pred,
    }
    print(f'[{i}] res:', db[i])
  gc.disable()

  with open(DB_FILE, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False)
  print(f'>> pAcc: {sum(rec["ok"] for rec in db.values()) / len(db):.5%}')

  color = []
  out_0 = []
  out_1 = []
  for idx, res in db.items():
    if type(res) == str: continue
    logits = res['logits']
    color.append(truth[int(idx)])
    out_0.append(logits[0])
    out_1.append(logits[1])
  plt.clf()
  plt.scatter(out_0, out_1, c=color)
  plt.xlabel('logits_0')
  plt.ylabel('logits_1')
  plt.suptitle(model_name)
  plt.show()


if __name__ == '__main__':
  stats()
