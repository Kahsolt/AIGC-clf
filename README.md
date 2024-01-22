# AIGC-clf

    Contest solution for distinguish AIGC images

----

Contest Page: [https://xihe.mindspore.cn/competition/mindcon23-aigc-img/0/introduction](https://xihe.mindspore.cn/competition/mindcon23-aigc-img/0/introduction)  
Team Name: 你这图保真吗  
Submission repo: [https://xihe.mindspore.cn/projects/kahsolt/AIGC-clf](https://xihe.mindspore.cn/projects/kahsolt/AIGC-clf)  
Final Score/Rank: 71%/7th, no award :( (refer to [SOLUTIONS.md](SOLUTIONS.md) for top solutions)

### Results

⚪ migrated pretrained apps

| method | pAcc | comment |
| :-: | :-: | :-: |
| cheaty | 99.897224% | detect image h/w==512 |
| [AI-generated-art-classifier](https://huggingface.co/spaces/artfan123/AI-generated-art-classifier) | 88.69476% | resnet18 clf |
| [AI-image-detector](https://huggingface.co/umm-maybe/AI-image-detector) | 78.82837% | swin clf |
| [sdxl-detector](https://huggingface.co/Organika/sdxl-detector) | 56.62898% | swin clf |
| [sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) | 70.7086% | aekl + clf by loss_diff |

⚪ finetuned apps

ℹ following apps are based on `AI-generated-art-classifier` and `sd-vae-ft-ema`

- run `python predict.py --app <app_name>`

| app |  input_size | pAcc | pAcc (`vote=5`) | pAcc (`vote=7`) | comment |
| :-: | :-: | :-: | :-: | :-: | :-: |
| resnet    | 224 | 85.71429% | 85.50874% | 85.09764% | migrated baseline |
| resnet_ft | 224 | 98.86948% | 99.07503% | 99.07503% | finetune |
| resnet_hf | 80  | 87.97533% | 93.01131% | 93.62795% | retrain from pretrained |
| aekl_clf  | 256 | 95.67456% | 95.88900% | 96.60843% | finetune from pretrained |

⚪ ensemble app

- run `python predict_ensemble.py --votes 7`

| app | pAcc  | pAcc (`vote=7`) |
| :-: | :-: | :-: |
| resnet_ft | 96.40288% | 97.22508% |
| resnet_hf | 89.31141% | 96.71120% |
| aekl_clf  | 96.50565% | 97.43063% |
| ensembled | 98.15005% | 99.38335% |


### Quickstart

⚪ run pretrained apps

- install [PyTorch](https://pytorch.org/get-started/locally/)
- `pip install -r requirements.txt`
- run `python predict.py --app <app_name>` and see `out/result.txt`

⚪ finetune the apps

- link the [contest dataset](https://xihe.mindspore.cn/datasets/drizzlezyk/ai-real/tree/image) to `data`
- run the following train scripts
  - `python train_resnet_ft.py`
  - `python train_resnet_hf.py`
  - `python train_aekl_clf.py`


#### references

- AI-detectors
  - [https://huggingface.co/Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector)
  - [https://huggingface.co/umm-maybe/AI-image-detector](https://huggingface.co/umm-maybe/AI-image-detector)
  - [https://huggingface.co/artfan123/resnet-18-finetuned-ai-art](https://huggingface.co/artfan123/resnet-18-finetuned-ai-art)
    - [https://huggingface.co/spaces/artfan123/AI-generated-art-classifier](https://huggingface.co/spaces/artfan123/AI-generated-art-classifier)
- SD VAEs
  - [https://huggingface.co/stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae)
  - [https://huggingface.co/stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
  - [https://huggingface.co/stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema)
  - [https://huggingface.co/stabilityai/sd-vae-ft-mse-original](https://huggingface.co/stabilityai/sd-vae-ft-mse-original)
- dataset
  - JourneyDB: [https://huggingface.co/datasets/JourneyDB/JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB)
- 比赛材料仓库
  - 训练样例: [https://source-xihe-mindspore.osinfra.cn/champagne11/mindcv_twoclass.git](https://source-xihe-mindspore.osinfra.cn/champagne11/mindcv_twoclass.git)
  - 数据样例: [https://source-xihe-mindspore.osinfra.cn/drizzlezyk/ai-real.git](https://source-xihe-mindspore.osinfra.cn/drizzlezyk/ai-real.git)
  - mindspore: [https://www.mindspore.cn/docs/zh-CN/r2.2/index.html](https://www.mindspore.cn/docs/zh-CN/r2.2/index.html)
  - mindcv: [https://mindspore-lab.github.io/mindcv/](https://mindspore-lab.github.io/mindcv/)

----
by Armit
2024/01/03
