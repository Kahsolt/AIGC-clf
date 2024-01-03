#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

from pathlib import Path
from argparse import ArgumentParser

from PIL import Image


def predict(args):
  cmp_fp = lambda fp: int(Path(fp).stem)
  fps = sorted(Path(args.test_data_path).iterdir(), key=cmp_fp)

  preds = []
  for fp in fps:
    img = Image.open(fp).convert('RGB')
    w, h = img.size
    if w == 512 or h == 512:
      preds.append(1)
    else:
      preds.append(0)

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
