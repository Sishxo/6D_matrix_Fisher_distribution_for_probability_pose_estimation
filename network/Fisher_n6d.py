import torch
import torch.nn as nn


class Fisher_n6d(nn.Module):
    def __init__(self, resnet_head, fisher_head, rot_head, batch_size):
        super(Fisher_n6d, self).__init__()
        self.base_net = resnet_head
        self.fisher_head = fisher_head
        #self.rot_head = rot_head
        self.batch_size = batch_size

    def forward(self, x, class_idx):
        #print(x.shape)
        features = self.base_net(x)
        fisher_output = self.fisher_head(features, class_idx)
        feat = features.view(x.shape[0], 32, 8, 8)  # feature=[bs, 32, 8, 8]
        feat = feat.repeat(1, 64, 1, 1)  # feature=[bs, 2048, 8, 8]
        # 6d normal vectors
        green_R_vec, red_R_vec = self.rot_head(feat)  # green_R_vec = [bs, 4]
        # normalization: green_R_normalized = [bs, 3]
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6) 
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = torch.sigmoid(green_R_vec[:, 0]) # f_green_R = [bs]
        f_red_R = torch.sigmoid(red_R_vec[:, 0])
        return fisher_output, p_green_R, p_red_R, f_green_R, f_red_R
