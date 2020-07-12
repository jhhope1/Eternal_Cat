import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
from DataLoad import PlaylistDataset
import DataLoad

def splited_loader(validation_ratio = 0.1, test_ratio = 0.1, random_seed = 10, batch_size = 128):
    train_transformed_dataset = PlaylistDataset(transform=transforms.Compose([
                                                DataLoad.Noise_p(0.5),
                                            ]))
    test_transformed_dataset = PlaylistDataset(transform=transforms.Compose([
                                                DataLoad.Noise_p(0.5),
                                                DataLoad.ToTensor()
                                            ]))

    num_train = len(train_transformed_dataset)
    indices = list(range(num_train))
    split0 = int(np.floor(validation_ratio * num_train)) 
    split1 = int(np.floor((validation_ratio+test_ratio) * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, test_idx, valid_idx = indices[split1:], indices[split0:split1], indices[:split0]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_transformed_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )

    valid_loader = torch.utils.data.DataLoader(
        test_transformed_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_transformed_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0
    )
    return (train_loader, valid_loader, test_loader)
