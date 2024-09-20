import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def normalize_vector(v, return_mag=False):
    B = v.shape[0]
    device = v.get_device()

    v_mag = torch.sqrt(v.pow(2).sum(1))
    v_mag = torch.max(v_mag, Variable(torch.FloatTensor([1e-8]).to(device)))
    v_mag = v_mag.view(B,1).expand(B, v.shape[1])
    v = v/v_mag

    if return_mag:
        return v, v_mag[:,0]
    else:
        return v

def cross_product(u, v):
    B = u.shape[0]

    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(B,1), j.view(B,1), k.view(B,1)), dim=1)
        
    return out

def ortho6d_to_SO3(ortho6d):
    x_raw = ortho6d[:,0:3]
    y_raw = ortho6d[:,3:6]
        
    x = normalize_vector(x_raw)
    z = cross_product(x,y_raw)
    z = normalize_vector(z)
    y = cross_product(z,x)
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)

    matrix = torch.cat((x,y,z), dim=2)

    return matrix

def SO3_to_ortho6d(R):
    B = R.shape[0] 

    ortho6d = R[:,:,0:2]
    ortho6d = torch.transpose(ortho6d, 1, 2)
    ortho6d = ortho6d.reshape(B,-1)

    return ortho6d

def create_se3_matrix(R, p):
    B = R.shape[0]

    se3_matrix = torch.zeros(B, 4, 4)
    se3_matrix[:, :3, :3] = R
    se3_matrix[:, :3, 3] = p
    se3_matrix[:, 3, 3] = 1.0

    return se3_matrix