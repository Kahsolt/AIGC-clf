#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

from argparse import ArgumentParser

from huggingface.resnet import get_resnet18_finetuned_ai_art, get_resnet18_finetuned_ai_art_ft, infer_resnet
from huggingface.aekl_clf import get_aekl_clf, infer_aekl_clf
from huggingface.utils import *
from utils import *

torch.backends.cudnn.benchmark = False

@torch.inference_mode()
def predict():
  app = 4
  if app == 0:
    model_func = get_resnet18_finetuned_ai_art
    infer_func = infer_resnet
    exp_name = 'resnet'
  elif app == 1:
    model_func = get_resnet18_finetuned_ai_art_ft
    infer_func = infer_resnet
    exp_name = 'resnet_ft'
  elif app == 2:
    model_func = get_aekl_clf
    infer_func = infer_aekl_clf
    exp_name = 'aekl_clf'
  print('exp_name:', exp_name)

  truth = load_truth()
  model = model_func().to(device)
  fps = get_test_fps()

  db_file = OUT_PATH / f'stats_{exp_name}.json'
  db = load_db(db_file)

  preds = []
  tot, ok = 0, 0
  pbar = tqdm(fps)
  for i, fp in enumerate(pbar):
    img = Image.open(fp).convert('RGB')
    logits, probs, pred = infer_func(model, img, debug=True)
    outpred = 1 - pred  # swap 0-1
    preds.append(outpred)   
    db[i] = {
      'logits': logits,
      'probs': probs,
      'pred': pred,
      'ok': truth[i] == outpred,
    }
    if i % 10 == 0: save_db(db, db_file)

    ok += pred == truth[i]
    tot += 1
    pbar.update()
    pbar.set_description_str(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  save_db(db, db_file)
  print(f'>> pAcc: {sum(rec["ok"] for rec in db.values()) / len(db):.5%}')

  out_fp = Path(args.output_path)
  out_fp.parent.mkdir(exist_ok=True, parents=True)
  with open(out_fp, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')

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
  parser = ArgumentParser()
  parser.add_argument('--model_path', default='model.ckpt')
  parser.add_argument('--test_data_path', default='./test')
  parser.add_argument('--output_path', default='./output/result.txt')
  args, _ = parser.parse_known_args()

  predict(args)
