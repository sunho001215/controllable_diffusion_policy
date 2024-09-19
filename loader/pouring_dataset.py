import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pickle

class Pouring(Dataset):
    def __init__(
            self, 
            root='./datasets/pouring_data',
            **kwargs
        ):
      
        self.traj_data_ = [] 
        self.labels_ = [] 

        for file_ in os.listdir(root):
            with open(os.path.join(root, file_), "rb") as f:
                data = pickle.load(f)
                traj = data['traj']
                traj = traj@np.array(
                        [[
                            [1., 0., 0., data['offset'][0]], 
                            [0., 1., 0., data['offset'][1]], 
                            [0., 0., 1., data['offset'][2]], 
                            [0., 0., 0., 1.]]])
                
                self.traj_data_.append(torch.tensor(traj, dtype=torch.float32).unsqueeze(0))
                self.labels_.append(torch.tensor(data['label']).unsqueeze(0))
                    
        self.traj_data_ = torch.cat(self.traj_data_, dim=0)
        self.labels_ = torch.cat(self.labels_, dim=0)
      
        print(f'Pouring dataset is ready; # of trajectories: {len(self.traj_data_)}')
            
    def __getitem__(self, idx):
        traj = self.traj_data_[idx] # (-, Len, 4, 4)
        labels = self.labels_[idx] # (-, 2) : (pouring style, direction)
        return traj, labels

    def __len__(self) -> int:
        return len(self.traj_data_)