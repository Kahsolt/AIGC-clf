#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

from argparse import ArgumentParser

import mindspore as ms


def predict(args):
  pass


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--model_path', default='model.ckpt')
  parser.add_argument('--test_data_path', default='./test')
  parser.add_argument('--output_path', default='./output/result.txt')
  args, _ = parser.parse_known_args()

  predict(args)
