#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

from huggingface.resnet import get_resnet18_finetuned_ai_art, infer_resnet
from huggingface.swin import get_AI_image_detector, get_xl_detector, infer_swin
from huggingface.utils import *
from utils import *


@torch.inference_mode()
def stats():
  app = 0
  if app == 0:
    model_func = get_resnet18_finetuned_ai_art
    infer_func = infer_resnet
    exp_name = 'resnet'
  elif app == 1:
    model_func = get_AI_image_detector
    infer_func = infer_swin
    exp_name = 'swin_sd'
  elif app == 2:
    model_func = get_xl_detector
    infer_func = infer_swin
    exp_name = 'swin_sdxl'

  truth = load_truth()
  model = model_func()

  db_file = OUT_PATH / f'stats_{exp_name}.json'
  db = load_db(db_file)
  fps = get_test_fps()

  for i, fp in enumerate(tqdm(fps)):
    if str(i) in db: continue
    img = Image.open(fp).convert('RGB')
    logits, probs, pred = infer_func(model, img, debug=True)
    db[i] = {
      'logits': logits,
      'probs': probs,
      'pred': pred,
      'ok': truth[i] == 1 - pred,   # swap 0-1
    }
    print(f'[{i}] res:', db[i])

  save_db(db, db_file)
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
  plt.scatter(out_0, out_1, c=color, cmap='bwr')
  plt.xlabel('logits_0')    # AI
  plt.ylabel('logits_1')    # real
  plt.suptitle(exp_name)
  plt.savefig(IMG_PATH / f'stats_{exp_name}.png', dpi=800)


if __name__ == '__main__':
  stats()
