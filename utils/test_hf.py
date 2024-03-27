import timm
import torch
from PIL import Image
import os
from torch import nn

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# image = Image.open("../datasets/husky2.jpg")

# model = timm.create_model(
#     'vit_base_patch16_224.augreg2_in21k_ft_in1k',
#     pretrained=True,
#     num_classes=0,  # remove classifier nn.Linear
#     pretrained_cfg_overlay=dict(file='/data0/sunshichu/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k/pytorch_model.bin'),
# )

# model = model.eval()

# data_config = timm.data.resolve_model_data_config(model)
# transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(image)  # output is (batch_size, num_features) shaped tensor

# print(output)

tensor = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
print(tensor.view(-1,9)[:,::4])