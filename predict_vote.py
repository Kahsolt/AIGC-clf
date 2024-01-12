#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/12 

from scipy.stats import mode
from predict import *

PATCH_SIZE = 224

transform_valid = T.Compose([
  T.RandomResizedCrop((PATCH_SIZE, PATCH_SIZE), scale=(0.8, 1.0)),
  T.RandomCrop((224, 224)),
  T.ToTensor(),
  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
transform_resnet_var = transform_valid


@torch.inference_mode()
def predict(args):
  app = 1
  if app == 0:
    transform_func = transform_resnet_var
    model_func = get_resnet18_finetuned_ai_art
    infer_func = infer_resnet
    exp_name = 'resnet'
  elif app == 1:
    transform_func = transform_resnet_var
    model_func = get_resnet18_finetuned_ai_art_ft
    infer_func = infer_resnet
    exp_name = 'resnet_ft'
  elif app == 2:
    transform_func = transform_aekl_clf
    model_func = get_aekl_clf
    infer_func = infer_aekl_clf
    exp_name = 'aekl_clf'
  else: raise ValueError(app)

  print('exp_name:', exp_name)
  truth = load_truth()
  model = model_func().to(device, dtype)
  fps = get_test_fps(args.test_data_path)

  preds = []
  tot, ok = 0, 0
  pbar = tqdm(fps)
  for i, fp in enumerate(pbar):
    img = Image.open(fp).convert('RGB')
    votes = []
    for _ in range(5):
      X = transform_func(img).to(device, dtype)
      pred = infer_func(model, X)
      votes.append(1 - pred) # swap 0-1
    outpred = mode(votes).mode
    preds.append(outpred)   

    ok += truth[i] == outpred
    tot += 1
    pbar.update()
    if i % 10 == 0:
      pbar.set_description_str(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  print(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  out_fp = Path(args.output_path)
  out_fp.parent.mkdir(exist_ok=True, parents=True)
  with open(out_fp, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')


if __name__ == '__main__':
  args = get_args()
  predict(args)
