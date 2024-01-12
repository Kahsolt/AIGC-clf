#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

import json
from pathlib import Path
from typing import *

from tqdm import tqdm
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from skimage.metrics import *
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent
HF_PATH = BASE_PATH / 'huggingface'
DATA_PATH = BASE_PATH / 'test'
DATA_FAKE_PATH = BASE_PATH / 'data' / 'imgs'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
OUT_PATH = BASE_PATH / 'output' ; OUT_PATH.mkdir(exist_ok=True)
TRUTH_FILE = OUT_PATH / 'result-ref.txt'
RESULT_FILE = OUT_PATH / 'result.txt'

npimg = ndarray


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

def to_ch_avg(x:ndarray) -> ndarray:
  return np.tile(x.mean(axis=-1, keepdims=True), (1, 1, 3))

def to_gray(im:npimg) -> npimg:
  return pil_to_npimg(npimg_to_pil(im).convert('L'))

def minmax_norm(dx:ndarray, vmin:int=None, vmax:int=None) -> npimg:
  vmin = vmin or dx.min()
  vmax = vmax or dx.max()
  out = (dx - vmin) / (vmax - vmin)
  return (out * 255).astype(np.uint8)

def npimg_diff(x:npimg, y:npimg) -> ndarray:
  return x.astype(np.int16) - y.astype(np.int16)

def npimg_abs_diff(x:npimg, y:npimg, name:str=None) -> npimg:
  d = np.abs(npimg_diff(x, y))
  if name:
    print(f'[{name}]')
    print('  Linf:', d.max() / 255)
    print('  L1:',  d.mean() / 255)
  return d
