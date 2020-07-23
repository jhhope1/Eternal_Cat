import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import transforms
import random
from DataLoad_2 import PlaylistDataset as PLDT_2
from DataLoad import PlaylistDataset as PLDTe_1
import DataLoad

def splited_loader(test_ratio = 0.1, random_seed = 10, batch_size = 128, noise_p = 0.5):
    transformed_dataset = PLDT_2(noise_p = noise_p)
    transformed_test_dataset = PLDTe_1(transform=transforms.Compose([
                                                DataLoad.add_plylst_meta(),
                                                DataLoad.Noise_p(0.5),
                                                DataLoad.add_meta(),
                                                DataLoad.ToTensor()
                                            ]))

    num_train = len(transformed_dataset)
    indices = list(range(num_train))
    split = int(np.floor(test_ratio * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        transformed_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        transformed_test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0
    )
    return (train_loader, test_loader)