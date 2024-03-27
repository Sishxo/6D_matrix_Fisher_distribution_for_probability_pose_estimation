import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import absl.flags as flags
FLAGS = flags.FLAGS
def get_vertical_rot_vec(c1, c2, y, z):
    ##  c1, c2 are weights
    ##  y, x are rotation vectors
    y = y.view(-1)
    z = z.view(-1)
    rot_x = torch.cross(y, z)
    rot_x = rot_x / (torch.norm(rot_x) + 1e-8)
    # cal angle between y and z
    y_z_cos = torch.sum(y * z)
    y_z_theta = torch.acos(y_z_cos)
    theta_2 = c1 / (c1 + c2) * (y_z_theta - math.pi / 2)
    theta_1 = c2 / (c1 + c2) * (y_z_theta - math.pi / 2)
    # first rotate y
    c = torch.cos(theta_1)
    s = torch.sin(theta_1)
    rotmat_y = torch.tensor([[rot_x[0]*rot_x[0]*(1-c)+c, rot_x[0]*rot_x[1]*(1-c)-rot_x[2]*s, rot_x[0]*rot_x[2]*(1-c)+rot_x[1]*s],
                             [rot_x[1]*rot_x[0]*(1-c)+rot_x[2]*s, rot_x[1]*rot_x[1]*(1-c)+c, rot_x[1]*rot_x[2]*(1-c)-rot_x[0]*s],
                             [rot_x[0]*rot_x[2]*(1-c)-rot_x[1]*s, rot_x[2]*rot_x[1]*(1-c)+rot_x[0]*s, rot_x[2]*rot_x[2]*(1-c)+c]]).to(y.device)
    new_y = torch.mm(rotmat_y, y.view(-1, 1))
    # then rotate z
    c = torch.cos(-theta_2)
    s = torch.sin(-theta_2)
    rotmat_z = torch.tensor([[rot_x[0] * rot_x[0] * (1 - c) + c, rot_x[0] * rot_x[1] * (1 - c) - rot_x[2] * s,
                              rot_x[0] * rot_x[2] * (1 - c) + rot_x[1] * s],
                             [rot_x[1] * rot_x[0] * (1 - c) + rot_x[2] * s, rot_x[1] * rot_x[1] * (1 - c) + c,
                              rot_x[1] * rot_x[2] * (1 - c) - rot_x[0] * s],
                             [rot_x[0] * rot_x[2] * (1 - c) - rot_x[1] * s,
                              rot_x[2] * rot_x[1] * (1 - c) + rot_x[0] * s, rot_x[2] * rot_x[2] * (1 - c) + c]]).to(z.device)

    new_z = torch.mm(rotmat_z, z.view(-1, 1))
    new_R=torch.stack((y,z,rot_x),dim=-1)
    return new_y.view(-1), new_z.view(-1), rot_x.view(-1),new_R

def get_rot_mat_y_first(y, x):
    # poses

    y = F.normalize(y, p=2, dim=-1)  # bx3
    z = torch.cross(x, y, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    x = torch.cross(y, z, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)

def get_rot_vec_vert_batch(c1, c2, x, y):
    bs = c1.shape[0]
    new_x = torch.zeroslike(x)
    new_y = torch.zeroslike(y)
    new_z = torch.zeroslike(x)
    new_R = torch.zeros((bs,3,3),dtype=torch.float32,device=x.device)
    for i in range(bs):
        new_x[i,...],new_y[i,...],new_z[i,...],new_R[i,...] = get_vertical_rot_vec(c1[i, ...], c2[i, ...], x[i, ...], y[i, ...])
    return new_x,new_y,new_z,new_R

def get_gt_v(batch_size, Rs, axis=2):
    bs = batch_size  # Rs = bs x 3 x 3

    corners = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.float).to(Rs.device) # corners = [3, 3]
    corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
    
    gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1) # gt_vec = [bs, 9]
    gt_green = gt_vec[:, 3:6] # gt_green = [bs, 3]
    gt_red = gt_vec[:, (6, 7, 8)] # gt_red = [bs, 3]
    
    return gt_green, gt_red

if __name__ == '__main__':
    g_R=torch.tensor([[0.3126, 0.0018, -0.9499],
            [0.7303, -0.6400, 0.2391],
            [-0.6074, -0.7684, -0.2014]], device='cuda:0')
    
    U,S,V = torch.svd(g_R)
    print(U,S,V)