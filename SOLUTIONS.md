### ranklist

| repo | score |
| :-: | :-: |
| [NicholasKX/ai_image_detector](https://xihe.mindspore.cn/projects/NicholasKX/ai_image_detector) | 0.953 |
| [Yuzi0123/AI-image-classification](https://xihe.mindspore.cn/projects/Yuzi0123/AI-image-classification) | 0.8792 |
| [Dexg/Discriminator](https://xihe.mindspore.cn/projects/Dexg/Discriminator) | 0.8344 |
| [zhouyifengCode/ai_real_pic_detection](https://xihe.mindspore.cn/projects/zhouyifengCode/ai_real_pic_detection) | 0.784 |
| [chixiaohang2023/AIGC_TWO_CLASSES](https://xihe.mindspore.cn/projects/chixiaohang2023/AIGC_TWO_CLASSES) | 0.7486 |
| [arance/ai2class](https://xihe.mindspore.cn/projects/arance/ai2class) | 0.7336 |
| [kahsolt/AIGC-clf](https://xihe.mindspore.cn/projects/kahsolt/AIGC-clf) | **0.71** |
| [jieXu/mindCon5-twoClass](https://xihe.mindspore.cn/projects/jieXu/mindCon5-twoClass) | 0.5096 |
| [FrankLiu/paddlepaddle](https://xihe.mindspore.cn/projects/FrankLiu/paddlepaddle) | 0.4992 |
| [codeer/cls](https://xihe.mindspore.cn/projects/codeer/cls) | 0.475 |


### solutions

⚪ NicholasKX/ai_image_detector

- model: resnet50
- optim: Adam[lr=1e-5~1e-7], epoch=30, bs=192
- transform
  - infer
    - Resize(224)
    - RandomCrop(224)
    - Norm[imagenet]

⚪ Yuzi0123/AI-image-classification

- model: cmt_small
- optim: Adamw[lr=2e-3~1e-5], epoch=50, bs=200
- transform:
  - train:
    - Resize(224)
    - Resize(224)
  - infer: ImagenetTransform

⚪ Dexg/Discriminator

- model: Unet/resnet50
- optim: Adam[lr=2e-5], epoch=100, bs=128
- data: [train=21796/valid=1121]

⚪ zhouyifengCode/ai_real_pic_detection

- model: mobilenet_v3_small
- transform
  - Resize(448)
  - Crop(0.875)
  - Norm[imagenet]

⚪ chixiaohang2023/AIGC_TWO_CLASSES

- model: xception
- transform: ImagenetTransform

⚪ arance/ai2class

- model: resnet50
- optim: Adam[lr=1e-3], epoch=100, bs=128
- data: imagenet_sd_small [train=63854/valid=3200]
- transform: ImagenetTransform

⚪ jieXu/mindCon5-twoClass

- model: googlenet
- optim: Adam[lr=1e-4], epoch=1, bs=128
- transform: None

⚪ FrankLiu/paddlepaddle

- model: densenet121
- optim: Adam[lr=1e-4], epoch=10, bs=16
- transform: ImagenetTransform

⚪ codeer/cls

- model: resnext50
- optim: SGD[lr=1e-4]
- data: imagenet_ai_0424_wukong
- transform
  - train
    - Resize(256)
    - RandomCrop(224)
    - RandomHorizontalFlip
    - Norm[imagenet]
  - infer
    - Resize(256)
    - CenterCrop(224)
    - Norm[imagenet]

----
by Armit
2024年1月22日
