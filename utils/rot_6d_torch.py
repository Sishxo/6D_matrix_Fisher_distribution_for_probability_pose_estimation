import torch
import numpy as np
from scipy.spatial.transform import Rotation
import time

def generate_random_matrix():
    r=Rotation.random()
    rotation_matrix_np=r.as_matrix()
    rotation_matrix_torch = torch.tensor(rotation_matrix_np,dtype=torch.float32)
    return rotation_matrix_torch.view(-1,3,3)

def torch_matrix_to_6d(mat):
    if not ((mat.shape[-1]==3 and mat.shape[-1]==3) or (mat.shape[-1]==9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(mat.shape[-1]))
    
    if(len(mat.shape)==4):
        mat_shape1=mat.shape[1]
        mat_3x3 = mat.view(-1,mat_shape1,3,3)
    else:
        mat_3x3 = mat.view(-1,3,3)
    r6d = torch.cat([mat_3x3[...,0],mat_3x3[...,1]],axis=-1)
    
    return r6d

def torch_6d_to_matrix(r6d):
    if not r6d.shape[-1] == 6:
        raise AttributeError("The last demension of the inputs in tf_rotation6d_to_matrix should be 6, \
            but found tensor with shape {}".format(r6d.shape[-1]))
    
    r6d = r6d.view(-1,6)
    x_raw = r6d[:,0:3]
    y_raw = r6d[:,3:6]
    
    x = x_raw/torch.norm(x_raw,p=2,dim=-1,keepdim=True)
    z = torch.cross(x,y_raw)
    z = z/torch.norm(z,p=2,dim=-1,keepdim=True)
    y = torch.cross(z,x)
    
    x=x.view(-1,3,1)
    y=y.view(-1,3,1)
    z=z.view(-1,3,1)
    matrix = torch.cat([x,y,z],axis=-1)
    
    return matrix

def torch_6d_to_matrix_as_formula(r6d):
    r6d = r6d.view(-1,6)
    x_raw = r6d[:,0:3]
    y_raw = r6d[:,3:6]
    
    x = x_raw/torch.norm(x_raw,p=2,dim=-1,keepdim=True)
    
    b2 = y_raw-torch.sum(torch.mul(x,y_raw),dim=1)*x
    y = b2/torch.norm(b2,p=2,dim=-1,keepdim=True)
    
    z = torch.cross(x,y)
    
    x=x.view(-1,3,1)
    y=y.view(-1,3,1)
    z=z.view(-1,3,1)
    matrix = torch.cat([x,y,z],axis=-1)
    
    return matrix
    
    
if __name__=='__main__':
    R=generate_random_matrix()
    
    print(R)

    R6d = torch_matrix_to_6d(R)
    
    # R = torch_6d_to_matrix(R6d)
    
    start=time.time()
    for i in range(100000):
        R_as_formula = torch_6d_to_matrix_as_formula(R6d)
    end=time.time()
    print(end-start)
    start=time.time()
    for i in range(100000):
        R = torch_6d_to_matrix(R6d)
    end=time.time()
    print(end-start)
