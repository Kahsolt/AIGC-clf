#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

# https://huggingface.co/artfan123/resnet-18-finetuned-ai-art

import numpy as np
from transformers import ImageClassificationPipeline, AutoImageProcessor, AutoModelForImageClassification
from transformers.models.convnext.image_processing_convnext import ConvNextImageProcessor
from transformers.models.resnet.configuration_resnet import ResNetConfig
from transformers.models.resnet.modeling_resnet import ResNetForImageClassification

processor: ConvNextImageProcessor = AutoImageProcessor.from_pretrained("artfan123/resnet-18-finetuned-ai-art")
model: ResNetForImageClassification = AutoModelForImageClassification.from_pretrained("artfan123/resnet-18-finetuned-ai-art")

processor.to_json_file('processor.json')
model.config.to_json_file('model.json')
np.savez('model.npz', **{k: v.cpu().numpy() for k, v in model.state_dict().items()})

with open('arch.txt', 'w', encoding='utf-8') as fh:
  fh.write(str(processor))
  fh.write('\n')
  fh.write('----')
  fh.write('\n')
  fh.write(str(model))

processor = ConvNextImageProcessor.from_json_file('processor.json')
model = ResNetForImageClassification(ResNetConfig.from_json_file('model.json'))
pipe = ImageClassificationPipeline(model=model, image_processor=processor)
