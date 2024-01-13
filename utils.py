#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

import json
from pathlib import Path
from typing import *

from tqdm import tqdm
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import seaborn as sns

BASE_PATH = Path(__file__).parent
HF_PATH = BASE_PATH / 'huggingface'
DATA_PATH = BASE_PATH / 'data'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
OUT_PATH = BASE_PATH / 'out' ; OUT_PATH.mkdir(exist_ok=True)
TRUTH_FILE = OUT_PATH / 'result-ref.txt'
RESULT_FILE = OUT_PATH / 'result.txt'

npimg = ndarray
mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def get_test_fps(dp:Path=DATA_PATH):
  cmp_fp = lambda fp: int(Path(fp).stem)
  fps = sorted(dp.iterdir(), key=cmp_fp)
  return fps


def load_preds(fp)-> list:
  with open(fp) as fh:
    lines = fh.read().strip().split('\n')
  return [int(ln) for ln in lines]

def load_truth() -> List[int]:
  return load_preds(TRUTH_FILE)


def load_db(fp:Path) -> Dict[str, Any]:
  if fp.exists():
    with open(fp, 'r', encoding='utf-8') as fh:
      db = json.load(fh)
  else:
    db = {}
  return db

def save_db(db:Dict[str, Any], fp:Path):
  with open(fp, 'w', encoding='utf-8') as fh:
    json.dump(db, fh, indent=2, ensure_ascii=False)


def load_img(fp:Path) -> PILImage:
  return Image.open(fp).convert('RGB')

def pil_to_npimg(img:PILImage) -> npimg:
  return np.asarray(img, dtype=np.uint8)

def npimg_to_pil(im:npimg) -> PILImage:
  assert im.dtype in [np.uint8, np.float32]
  if im.dtype == np.float32:
    assert 0.0 <= im.min() and im.max() <= 1.0
  return Image.fromarray(im)

def minmax_norm(dx:ndarray, vmin:int=None, vmax:int=None) -> npimg:
  vmin = vmin or dx.min()
  vmax = vmax or dx.max()
  out = (dx - vmin) / (vmax - vmin)
  return (out * 255).astype(np.uint8)

def npimg_diff(x:npimg, y:npimg) -> ndarray:
  return x.astype(np.int16) - y.astype(np.int16)

def to_hifreq(img:PILImage) -> ndarray:
  img_lo = img.filter(ImageFilter.GaussianBlur(3))
  im_lo = pil_to_npimg(img_lo)
  im = pil_to_npimg(img)
  im_hi = npimg_diff(im, im_lo)   # int16
  return (im_hi / 255.0).astype(np.float32)
