'''
This file is based on:
    https://github.com/ermongroup/tile2vec/blob/master/src/datasets.py

----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2021, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import scipy.io


class SceneTripletsDataset(Dataset):

    def __init__(self, scene_dir, transform=None):
        self.scene_dir = scene_dir
        mat = scipy.io.loadmat(scene_dir+ 'meta.mat')
        files = mat['files']
        files = np.squeeze(files)
        labels = mat['labels']
        labels = np.squeeze(labels)
        labels_groups = mat['labels_groups']
        labels_groups = np.squeeze(labels_groups)
        labels_groups = [labels_groups[i][0] for i in range(labels_groups.shape[0])]
        classes = np.unique(labels)
        index_per_group = []
        for i in classes:
            index_per_group.append([index for index,value in enumerate(labels) if value == i])
        
        self.index_per_group = index_per_group
        self.files = [files[i][0] for i in range(files.shape[0])]
        self.graphID = labels
        self.graphID_unique = classes
        self.graphID_groups = labels_groups
        self.transform = transform


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load anchor
        a = resize(rgb2gray(io.imread(self.scene_dir + self.files[idx] )),(64,64)).reshape((1,64,64))
        a_graphID = self.graphID[idx]
        
        # Load Randomly picked neighbor
        possible_index = self.index_per_group[a_graphID]
        idx_neighbor = np.random.choice(possible_index)
        while idx_neighbor == idx:
            idx_neighbor = np.random.choice(possible_index)
        n = resize(rgb2gray(io.imread(self.scene_dir + self.files[idx_neighbor] )),(64,64)).reshape((1,64,64))

        # Load Randomly picked distant
        possible_graphID = np.setdiff1d(self.graphID_unique,a_graphID)
        anti_graphID = np.random.choice(possible_graphID)
        possible_index = self.index_per_group[anti_graphID]
        idx_distant = np.random.choice(possible_index)
        d = resize(rgb2gray(io.imread(self.scene_dir + self.files[idx_distant] )),(64,64)).reshape((1,64,64))

        sample = {'anchor': a, 'neighbor': n, 'distant': d, 'graphID_a': a_graphID.astype(np.int32), 'graphID_p': a_graphID.astype(np.int32) , 'graphID_d': anti_graphID.astype(np.int32)}
        if self.transform:
            sample = self.transform(sample)
        return sample


### TRANSFORMS ###
class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, d = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            torch.from_numpy(sample['distant']).float())
        sample = {'anchor': a, 'neighbor': n, 'distant': d, 'graphID_a': sample['graphID_a'], 'graphID_p': sample['graphID_p'], 'graphID_d': sample['graphID_d']}
        return sample




def triplet_dataloader( scene_dir, batch_size=4, shuffle=True, num_workers=4):
    """
    Returns a DataLoader with bw infrastructure images.
    """
    transform_list = []
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = SceneTripletsDataset(scene_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers)
    return dataloader

    