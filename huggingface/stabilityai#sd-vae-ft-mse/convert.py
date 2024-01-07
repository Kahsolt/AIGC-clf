#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

# https://huggingface.co/stabilityai/sd-vae-ft-mse

import numpy as np
from diffusers.models import AutoencoderKL

model: AutoencoderKL = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

model.to_json_file('model.json')
np.savez('model.npz', **{k: v.cpu().numpy() for k, v in model.state_dict().items()})

with open('arch.txt', 'w', encoding='utf-8') as fh:
  fh.write(str(model))

model = AutoencoderKL.from_config('model.json')
