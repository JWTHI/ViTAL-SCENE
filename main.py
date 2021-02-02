'''
This file is based on:
    https://github.com/ermongroup/tile2vec/blob/master/examples/Example%202%20-%20Train%20Tile2Vec%20from%20scratch.ipynb

----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2021, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
import warnings
warnings.filterwarnings("ignore", message='Tensorflow not installed; ParametricUMAP', category=UserWarning)

import os
import os.path
from time import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import scipy.io

import umap

from src.datasets import SceneTripletsDataset, triplet_dataloader
from src.scenenet import make_scenenet
from src.training import  train_triplet_epoch

# DEFS--------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SCENE_DIR = BASE_DIR+'\\Data\\'
MODEL_DIR = BASE_DIR+'\\models\\'
MODEL_NAME = 'SceneNet_LS_50_Epoch200'

NUM_WORKERS = 4

Z_DIM = 50
MARGIN = 1

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
L2 = 0.01
CLIP = 100

GENERATE_LATENT_VIS = True
PLOT_LOSS_CURVE = True
PRINT_EVERY = 1000
SAVE_MODEL = True
LOAD_MODEL = False
    
BATCH_SIZE_PREDICT = 50


def main():
    # Environment stuff
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = torch.cuda.is_available()

    # Set up dataloader
    dataloader = triplet_dataloader(SCENE_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print('Dataloader set up complete.')#

    # Set up scenenet
    SceneNet = make_scenenet(z_dim=Z_DIM)
    SceneNet.train()
    if cuda: SceneNet.cuda()
    optimizer = optim.Adam(SceneNet.parameters(), lr=LR, betas=(0.5, 0.999))
    print('SceneNet set up complete.')
    
    if GENERATE_LATENT_VIS or LOAD_MODEL:
        # Load images for latent representation generation
        mat = scipy.io.loadmat(SCENE_DIR+ 'meta.mat')
        files = mat['files']
        files = np.squeeze(files)
        file_list = [files[i][0] for i in range(files.shape[0])]
        labels = mat['labels']
        labels = np.squeeze(labels)
        x = np.array( [resize(rgb2gray(io.imread(SCENE_DIR+fname)),(64,64)).reshape((1,64,64)) for fname in file_list] )

    if not LOAD_MODEL:
        
        if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
        if not os.path.exists(BASE_DIR+'\\results\\') and GENERATE_LATENT_VIS: os.makedirs(BASE_DIR+'\\results\\')

        loss_epoch = []
        epoch_vector = []
        t0 = time()
        results_fn = os.path.join(BASE_DIR, MODEL_NAME + '.txt')
        #---------------------------------------------------------
        # TRAINING
        #---------------------------------------------------------
        with open(results_fn, 'w') as file:
            print('Begin training.................')
            if PLOT_LOSS_CURVE:
                fig_epoch = plt.figure(1)
            for epoch in range(0, EPOCHS):
                # Train the Epoch
                (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
                    SceneNet, cuda, dataloader, optimizer, epoch+1, margin=MARGIN, l2=L2,
                    print_every=PRINT_EVERY, t0=t0, clip=CLIP)
                
                if PLOT_LOSS_CURVE:
                    # Visualize the loss
                    loss_epoch.append(avg_loss)
                    epoch_vector.append(epoch)
                    plt.figure(1)
                    plt.cla()
                    plt.plot(epoch_vector,loss_epoch,color='k')
                    plt.pause(0.05)
                    plt.figure(1)
                    plt.savefig(MODEL_NAME + '.pdf')

                if GENERATE_LATENT_VIS:
                    # Visualize the latent Space
                    SceneNet.eval()
                    # Get z for all images
                    z_all = np.zeros([0,Z_DIM])
                    for i in range(0,x.shape[0],BATCH_SIZE_PREDICT):
                        z = SceneNet.encode(torch.from_numpy(x[i:min(x.shape[0],i+BATCH_SIZE_PREDICT),:,:,:]).float().cuda())
                        z = (Variable(z).data).cpu().numpy()
                        z_all = np.append(z_all,z,axis=0)
    
                    if Z_DIM > 2:
                        reducer = umap.UMAP()
                        embedding = reducer.fit_transform(z_all)
                        plt.figure(2)
                        plt.cla()
                        plt.scatter(embedding[:, 0],embedding[:, 1],c=labels, cmap='viridis', s=5)
                        plt.savefig('results\\'+str(epoch)+'.pdf')
                        plt.close()
                    else:
                        embedding = z_all
                        plt.figure(2)
                        plt.cla()
                        plt.scatter(embedding[:, 0],embedding[:, 1],c=labels, cmap='viridis', s=5)
                        plt.savefig('results\\'+str(epoch)+'.pdf')
                        plt.close()
    
                    # Visualize reconstruction
                    ids_rand = np.random.randint(0,x.shape[0],20)
                    x_rand = x[ids_rand,:,:,:]
                    x_pred = SceneNet(torch.from_numpy(x_rand).float().cuda())
                    x_pred = (Variable(x_pred).data).cpu().numpy()
                    fig, axes = plt.subplots(4, 10,num=3)
                    print(x_rand.shape)
                    base_count = 0
                    for row_idx in np.arange(0,4,2):
                        for idx in np.arange(10):
                            axes[row_idx  ,idx].imshow(np.squeeze(x_rand[base_count+idx,:,:,:]))
                            axes[row_idx  ,idx].axis('off')
                            axes[row_idx+1,idx].imshow(np.squeeze(x_pred[base_count+idx,:,:,:]))
                            axes[row_idx+1,idx].axis('off')
                        base_count =+10
                    plt.savefig('results\\recon_'+str(epoch)+'.pdf')
                    plt.close()
    
                    SceneNet.train()

        # Save model after last epoch
        if SAVE_MODEL:
            model_fn = os.path.join(MODEL_DIR, MODEL_NAME + '.ckpt')
            torch.save(SceneNet.state_dict(), model_fn)
    else:
        SceneNet.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME + '.ckpt')))
        SceneNet.eval()

        z_all = np.zeros([0,z_dim])

        for i in range(0,x.shape[0],BATCH_SIZE_PREDICT):
            z = SceneNet(torch.from_numpy(x[i:min(x.shape[0],i+BATCH_SIZE_PREDICT),:,:,:]).float().cuda())
            z = (Variable(z).data).cpu().numpy()
            z_all = np.append(z_all,z,axis=0)
        
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(z_all)
        fig,ax = plt.subplots()
        ax.scatter(embedding[:, 0],embedding[:, 1],c=labels, cmap='viridis', s=5)
        fig.savefig('results/PredictOnly.pdf')
    
if __name__ == '__main__':
    main()
