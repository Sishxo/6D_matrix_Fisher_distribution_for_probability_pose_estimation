import sys
sys.path.append("..")

from network.rot_head import RotHeadNet
from network.resnet import resnet50, resnet101, ResnetHead
from network.Fisher_n6d import Fisher_n6d
import torch


base = resnet101(pretrained=True, progress=True)
fisher_head = ResnetHead(base.output_size, 13, 32, 512, 9)
rot_head = RotHeadNet(base.output_size)
model = Fisher_n6d(base,fisher_head,rot_head)

tensor = torch.randn(1,3,224,224)
fisher_output,green_R_vec, red_R_vec = model.forward(tensor,torch.tensor([1]))

print(fisher_output.shape)
print(green_R_vec.shape)
print(red_R_vec.shape)

