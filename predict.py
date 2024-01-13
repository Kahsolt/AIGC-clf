#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/07 

from argparse import ArgumentParser

from models.resnet import *
from models.aekl_clf import *
from models.utils import *
from utils import *

torch.backends.cudnn.benchmark = False

VALID_APP_NAMES = [name[len('get_app_'):] for name in globals() if name.startswith('get_app_')]


def transform_resnet_hf(img:PILImage) -> ndarray:
  im_hi = to_hifreq(img)
  im_hi = im_to_tensor(im_hi)
  im_hi = transform_highfreq(im_hi)
  return im_hi


@torch.inference_mode()
def predict(args, app_name:str='resnet'):
  is_vote = args.votes > 0
  app_name = app_name.replace('-', '_')
  model_func = globals()[f'get_app_{app_name}']

  if app_name == 'resnet':
    from models.resnet import transform as transform_func
    infer_func = infer_clf_batch if is_vote else infer_clf
  elif app_name == 'resnet_ft':
    from models.resnet import transform as transform_func
    infer_func = infer_clf_batch if is_vote else infer_clf
  elif app_name == 'resnet_hf':
    transform_func = transform_resnet_hf
    infer_func = infer_clf_batch if is_vote else infer_clf
  elif app_name == 'aekl_clf':
    from models.aekl_clf import transform as transform_func
    infer_func = infer_clf_batch if is_vote else infer_clf
  else: raise ValueError(app_name)

  print('app_name:', app_name)
  truth = load_truth()
  model = model_func().to(device, dtype)
  fps = get_test_fps(args.test_data_path)

  if not is_vote:
    db_file = OUT_PATH / f'stats_{app_name}.json'
    db = {}

  preds = []
  tot, ok = 0, 0
  pbar = tqdm(fps)
  for i, fp in enumerate(pbar):
    img = Image.open(fp).convert('RGB')

    if is_vote:
      X = torch.stack([transform_func(img) for _ in range(args.votes)], dim=0).to(device, dtype)
      votes = infer_func(model, X)
      mos = 1 - mean(votes)     # swap 0-1
      outpred = int(mos > args.thresh)
    else:
      X = transform_func(img).to(device, dtype)
      logits, probs, pred = infer_func(model, X, debug=True)
      outpred = 1 - pred  # swap 0-1
      db[i] = {
        'logits': logits,
        'probs': probs,
        'pred': pred,
        'ok': truth[i] == outpred,
      }

    preds.append(outpred)  
    ok += truth[i] == outpred
    tot += 1
    pbar.update()
    if i % 10 == 0:
      if not is_vote:save_db(db, db_file)
      pbar.set_description_str(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  if not is_vote:save_db(db, db_file)
  print(f'>> pAcc: {ok} / {tot} = {ok / tot:.5%}')

  out_fp = Path(args.output_path)
  out_fp.parent.mkdir(exist_ok=True, parents=True)
  with open(out_fp, 'w', encoding='utf-8') as fh:
    for p in preds:
      fh.write(f'{p}\n')

  if not is_vote:
    color = []
    out_0, out_1 = [], []
    for idx, res in db.items():
      if type(res) == str: continue
      logits = res['logits']
      color.append(truth[int(idx)])
      out_0.append(logits[0])
      out_1.append(logits[1])
    plt.clf()
    plt.scatter(out_0, out_1, s=1, c=color, cmap='bwr')
    plt.xlabel('AI')    # logits 0
    plt.ylabel('real')  # logits 1
    plt.suptitle(app_name)
    plt.savefig(IMG_PATH / f'stats_{app_name}.png', dpi=800)


def get_args():
  parser = ArgumentParser()
  parser.add_argument('--model_path',     type=Path, default='model.ckpt')
  parser.add_argument('--test_data_path', type=Path, default='./data')
  parser.add_argument('--output_path',    type=Path, default='./out/result.txt')
  parser.add_argument('--app',   default='resnet', choices=VALID_APP_NAMES)
  parser.add_argument('--votes', type=int, default=-1)
  parser.add_argument('--thresh', type=float, default=0.5)
  args, _ = parser.parse_known_args()
  return args


if __name__ == '__main__':
  args = get_args()
  predict(args, args.app)
