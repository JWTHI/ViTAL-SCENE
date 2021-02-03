'''
This file is based on:
    https://github.com/ermongroup/tile2vec/blob/master/src/tilenet.py
    
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2021, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from src.vit_pytorch import ViT


class SceneNet(nn.Module):
    def __init__(self, z_dim=50):
        super(SceneNet, self).__init__()
        self.z_dim = z_dim
        
        self.ViT = ViT(
            image_size = 64,
            patch_size = 8,
            z_dim = z_dim,
            dim = 256,
            depth = 20,
            heads = 16,
            mlp_dim = 128, 
            channels = 1
            )

        self.layerDec1 = nn.ConvTranspose2d(self.z_dim, 32,6,6)
        self.layerDec2 = nn.ConvTranspose2d(32, 64,4,2)
        self.layerDec3 = nn.ConvTranspose2d(64, 64,4,2)
        self.layerDec4 = nn.ConvTranspose2d(64, 1,6,2)
        

    def encode(self, x):
        z = self.ViT.forward(x)
        return z
    
    def decode(self, z):
        x = z.view((z.size(0),z.size(1),1,1))
        x = self.layerDec1(x)
        x = F.relu(x)
        x = self.layerDec2(x)
        x = F.relu(x)
        x = self.layerDec3(x)
        x = F.relu(x)
        x = self.layerDec4(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x 

    def triplet_loss(self, z_p, z_n, z_d, graphID_a, graphID_p, graphID_d , margin=0.1, l2=0):
        
        l_n = ((z_p - z_n) ** 2).sum(dim=1)

        # Find all negatives per anchor
        all_graphID = torch.cat((graphID_a,graphID_p,graphID_d),0)
        all_data = torch.cat((z_p,z_n,z_d),0)
        negatives_idx = torch.zeros(z_d.shape[0],dtype=torch.long)
        for idx,i in enumerate(graphID_a):
            # Find all negative anchors
            logical_index = torch.logical_not(torch.eq(all_graphID,i))
            anchor_stack = z_p[idx,:]
            a_negativs = all_data[logical_index,:]
            anchor_stack = anchor_stack.repeat(a_negativs.shape[0],1)

            # Calc distances of anchors to a_negatives
            distance_2_anchor = ((anchor_stack - a_negativs) ** 2).sum(dim=1)
            mask = (distance_2_anchor>l_n[idx]) & (distance_2_anchor<=l_n[idx]+margin)
            a_negativs = a_negativs[mask,:]
            if len(a_negativs)>0:
                negatives_idx[idx] = torch.randint(0,a_negativs.shape[0],(1,))
                z_d[idx,:] = a_negativs[negatives_idx[idx],:]
            else:
                negatives_idx[idx] = -1

        l_d = - ((z_p - z_d) ** 2).sum(dim=1)
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        for i,v in enumerate(negatives_idx):
            if v ==-1:
                loss[i] = 0
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def reconstruction_loss(self,x,x_pred):
        l = ((x - x_pred) ** 2).sum(dim=(1,2,3))
        l = torch.mean(l)
        return l

    def loss(self, patch, neighbor, distant, graphID_a, graphID_p, graphID_d , margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        patch_pred = self.decode(z_p)
        loss_trip, l_n, l_d, l_nd = self.triplet_loss(z_p, z_n, z_d,  graphID_a, graphID_p, graphID_d , margin=margin, l2=l2)
        reconstruction_loss = self.reconstruction_loss(patch,patch_pred)
        loss = loss_trip+reconstruction_loss
        return loss, l_n, l_d, l_nd,loss_trip,reconstruction_loss


def make_scenenet(z_dim=50):
    """
    Returns a SceneNet, providing a latent representation for road infrastructure images
    """
    return SceneNet(z_dim=z_dim)

