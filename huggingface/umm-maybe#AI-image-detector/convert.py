#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

# https://huggingface.co/umm-maybe/AI-image-detector

import numpy as np
from transformers.models.swin.configuration_swin import SwinConfig
from transformers.models.swin.modeling_swin import SwinForImageClassification

model: SwinForImageClassification = SwinForImageClassification.from_pretrained("umm-maybe/AI-image-detector")
model.config.to_json_file('model.json')
np.savez('model.npz', **{k: v.cpu().numpy() for k, v in model.state_dict().items()})

with open('arch.txt', 'w', encoding='utf-8') as fh:
  fh.write(str(model))

model = SwinForImageClassification(SwinConfig.from_json_file('model.json'))
