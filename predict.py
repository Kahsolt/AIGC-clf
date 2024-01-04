#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm

from models import *

REF_FILE = './output/result-ref.txt'

def load_preds(fp)-> list:
  with open(fp) as fh:
    lines = fh.read().strip().split('\n')
  return [int(ln) for ln in lines]


def predict(args):
  app = 2
  if app == 0:
    model_func = get_resnet18_finetuned_ai_art
    infer_func = infer_resnet18
  elif app == 1:
    model_func = get_AI_image_detector
    infer_func = infer_swin
  elif app == 2:
    model_func = get_xl_detector
    infer_func = infer_swin

  model = model_func()
  truth = load_preds(REF_FILE)

  cmp_fp = lambda fp: int(Path(fp).stem)
  fps = sorted(Path(args.test_data_path).iterdir(), key=cmp_fp)

  preds = []
  tot, ok = 0, 0
  pbar = tqdm(fps)
  for i, fp in enumerate(pbar):
    img = Image.open(fp).convert('RGB')
    pred = infer_func(model, img)
    preds.append(pred)

    ok += pred == truth[i]
    tot += 1
    pbar.update()
    pbar.set_description_str(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  print(f'pAcc: {ok} / {tot} = {ok / tot:.5%}')

  out_fp = Path(args.output_path)
  out_fp.parent.mkdir(exist_ok=True, parents=True)
  with open(out_fp, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--model_path', default='model.ckpt')
  parser.add_argument('--test_data_path', default='./test')
  parser.add_argument('--output_path', default='./output/result.txt')
  args, _ = parser.parse_known_args()

  predict(args)
