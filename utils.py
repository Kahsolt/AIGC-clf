#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/09 

import json
from pathlib import Path
from typing import *

from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = Path(__file__).parent
HF_PATH = BASE_PATH / 'huggingface'
DATA_PATH = BASE_PATH / 'test'
DATA_FAKE_PATH = BASE_PATH / 'data' / 'imgs'
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)
OUT_PATH = BASE_PATH / 'output' ; OUT_PATH.mkdir(exist_ok=True)
TRUTH_FILE = OUT_PATH / 'result-ref.txt'
RESULT_FILE = OUT_PATH / 'result.txt'


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
