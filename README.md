# AIGC-clf

    Contest solution for distinguish AIGC images

----

比赛主页: [https://xihe.mindspore.cn/competition/mindcon23-aigc-img/0/introduction](https://xihe.mindspore.cn/competition/mindcon23-aigc-img/0/introduction)  
Team Name: 你这图保真吗  


### Results

| method | pAcc | comment |
| :-: | :-: | :-: |
| cheaty | 99.897224% | detect image h/w==512 |
| [AI-generated-art-classifier](https://huggingface.co/spaces/artfan123/AI-generated-art-classifier) | 88.69476% | resnet18 clf |
| [AI-image-detector](https://huggingface.co/umm-maybe/AI-image-detector) | 78.82837% | swin clf |
| [sdxl-detector](https://huggingface.co/Organika/sdxl-detector) | 56.62898% | swin clf |
| [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) | 70.7086% | aekl + clf |

⚪ resnet18 finetuned (based on `AI-generated-art-classifier`)

| dtype | pAcc | comment |
| :-: | :-: | :-: |
| fp32 | 99.79445% | overfit all data |
| fp32 | 89.20863% | split 7:3 |
| fp16 | 90.54471% | split 7:3 |
| fp16 | 99.08257% | split 7:3 |
| bf16 | 89.92806% | split 7:3 |

⚪ aekl-clf finetuned (based on `sd-vae-ft-ema`)

| dtype | pAcc | comment |
| :-: | :-: | :-: |
| fp16 | 98.00000% | split 7:3 |


### Quickstart

- `conda create -n ms python==3.9 & conda activate ms`
- install [mindspore](https://github.com/mindspore-ai/mindspore#installation) and [mindcv](https://mindspore-lab.github.io/mindcv/installation/)
- run `predict.py`, see `output/result.txt`


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
