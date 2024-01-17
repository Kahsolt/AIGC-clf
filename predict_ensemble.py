#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

from predict import *

from models.resnet import transform as transform_func_resnet_ft
from models.resnet import transform_highfreq as transform_func_resnet_hf
from models.aekl_clf import transform as transform_func_aekl_clf


@torch.inference_mode()
def predict(args):
  truth = load_truth()
  fps = get_test_fps(DATA_PATH)
  models = [
    get_app_resnet_ft().to(device, dtype),
    get_app_resnet_hf().to(device, dtype),
    get_app_aekl_clf() .to(device, dtype),
  ]
  transform_funcs = [
    transform_func_resnet_ft,
    transform_func_resnet_hf,
    transform_func_aekl_clf
  ]

  preds = []
  tot, ok = 0, 0
  pbar = tqdm(fps)
  for i, fp in enumerate(pbar):
    img = Image.open(fp).convert('RGB')

    votes2 = []
    for model, transform_func in zip(models, transform_funcs):
      X = torch.stack([transform_func(img) for _ in range(args.votes)], dim=0).to(device, dtype)
      votes = infer_clf_batch(model, X)
      mos = 1 - mean(votes)     # swap 0-1
      votes2.append(int(mos > args.thresh))

    pred = mean(votes2) > args.thresh
    preds.append(pred)  
    ok += truth[i] == pred
    tot += 1
    pbar.update()
    if i % 10 == 0:
      pbar.set_description_str(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  print(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  with open(RESULT_FILE, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--votes', type=int, default=7)
  parser.add_argument('--thresh', type=float, default=0.5)
  args, _ = parser.parse_known_args()

  predict(args)
