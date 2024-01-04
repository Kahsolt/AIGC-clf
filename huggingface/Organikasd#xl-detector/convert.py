#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

# https://huggingface.co/Organika/sdxl-detector

import numpy as np
from transformers import ImageClassificationPipeline, AutoImageProcessor, AutoModelForImageClassification
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.models.swin.configuration_swin import SwinConfig
from transformers.models.swin.modeling_swin import SwinForImageClassification

processor: ViTImageProcessor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
model: SwinForImageClassification = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

processor.to_json_file('processor.json')
model.config.to_json_file('model.json')

with open('arch.txt', 'w', encoding='utf-8') as fh:
  fh.write(str(processor))
  fh.write('\n')
  fh.write('----')
  fh.write('\n')
  fh.write(str(model))

weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
np.savez('model.npz', **weights)

processor = ViTImageProcessor.from_json_file('processor.json')
model = SwinForImageClassification(SwinConfig.from_json_file('model.json'))
pipe = ImageClassificationPipeline(model=model, image_processor=processor)
