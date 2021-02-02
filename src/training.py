'''
This file is based on:
    https://github.com/ermongroup/tile2vec/blob/master/src/training.py
    
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2021, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
from time import time

import torch
from torch.autograd import Variable


def prep_triplets(triplets, cuda):
    """
    Takes a batch of triplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    a, n, d, graphID_a, graphID_p, graphID_d = (Variable(triplets['anchor']), Variable(triplets['neighbor']), Variable(triplets['distant']), Variable(triplets['graphID_a']), Variable(triplets['graphID_p']), Variable(triplets['graphID_d']))
    
    if cuda:
    	a, n, d = (a.cuda(), n.cuda(), d.cuda())
    return (a, n, d, graphID_a, graphID_p, graphID_d)

def train_triplet_epoch(model, cuda, dataloader, optimizer, epoch, margin=1,
    l2=0, print_every=100, t0=None,clip=100):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    model.train()

    if t0 is None:
        t0 = time.time()
    sum_loss, sum_l_n, sum_l_d, sum_l_nd,sum_loss_trip,sum_loss_recon = (0, 0, 0, 0,0,0)
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0.0

    #------------------------------------------
    # Training
    #------------------------------------------
    for idx, triplets in enumerate(dataloader):
        # Optimize for a batch
        p, n, d, graphID_a, graphID_p, graphID_d = prep_triplets(triplets, cuda)
        optimizer.zero_grad()
        loss, l_n, l_d, l_nd,loss_trip,reconstruction_loss = model.loss(p, n, d, graphID_a, graphID_p, graphID_d ,margin=margin, l2=l2)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        # Gather and Print results
        sum_loss += loss.data
        sum_l_n += l_n.data
        sum_l_d += l_d.data
        sum_l_nd += l_nd.data
        sum_loss_trip += loss_trip.data
        sum_loss_recon += reconstruction_loss.data
        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, sum_loss/(idx + 1)))
    
    # Print results
    avg_loss = sum_loss / n_batches
    avg_l_n = sum_l_n / n_batches
    avg_l_d = sum_l_d / n_batches
    avg_l_nd = sum_l_nd / n_batches
    avg_l_t = sum_loss_trip / n_batches
    avg_l_r = sum_loss_recon / n_batches
    print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
    print('  Average loss: {:0.4f}'.format(avg_loss))
    print('  Average l_n: {:0.4f}'.format(avg_l_n))
    print('  Average l_d: {:0.4f}'.format(avg_l_d))
    print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))
    print('  Average l_trip: {:0.4f}'.format(avg_l_t))
    print('  Average l_recon: {:0.4f}\n'.format(avg_l_r))
    return (avg_loss, avg_l_n, avg_l_d, avg_l_nd)