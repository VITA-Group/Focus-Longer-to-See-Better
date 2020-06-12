import numpy as np
import matplotlib.pyplot as plt
import os
#import cv2
import random
import time

import torch.nn.functional as F
import torch
from matplotlib.lines import Line2D
from preprocess import mean, std
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

normalize = transforms.Normalize(mean=mean,
                                 std=std)

def get_acc_from_logits(logits, labels):
    preds = torch.argmax(logits,1)
    return torch.sum(preds==labels).item()

def save_accuracy_figure(train_acc, test_acc, global_branch_acc, patch_branch_acc, savedir,  test_freq=2):
    plt.figure()
    try:
        plt.plot([i for i in range(0, len(train_acc), test_freq)], test_acc, color='r', label='Test Acc')
    except:
        print("Exception in plotting testing accuracy\n")
        plt.plot(test_acc, color='r', label='Test Acc')
    epochs = [i for i in range(len(train_acc))]
    plt.plot(epochs, train_acc, color='b', label='Train Acc')
    plt.plot(epochs, global_branch_acc, color='g', label='global_branch_acc')
    plt.plot(epochs, patch_branch_acc, color='y', label='patch_branch_acc')   

    plt.xlabel('epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.title("Accuracy Plot")
    plt.savefig(os.path.join(savedir,'acc'+'.png'))

def save_loss_figure(img_xent_loss, patch_xent_loss, savedir):
    plt.figure()
    plt.plot(img_xent_loss, color='r', label='img xent loss')
    plt.plot(patch_xent_loss, color='b', label='patch xent loss')

    plt.xlabel('epochs')
    plt.ylabel('Training Losses')
    plt.legend()
    plt.title("Training Loss Plot")
    plt.savefig(os.path.join(savedir,'loss'+'.png'))

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

if __name__ == '__main__':
    T = transforms.Compose([
    transforms.Resize(size=(448, 448)),
    transforms.ToTensor() #DO NOT NORMALIZE HERE
    ])
    
    dataset = CNN_3D_Loader('Train', 224, 4, "./448_train_patches/train_patches.npy", "./448_test_patches/test_patches.npy", '/ssd2/tianlong/CUB200/CUB_200_2011/train', T)
    dataloader =  DataLoader(dataset=dataset, batch_size=4, num_workers=2, shuffle=False)

    start = time.time()
    for step, (img, labels, patches) in enumerate(dataloader):
        print("img {} patches {}".format(img.shape, patches.shape))
        pass
    print("Time taken: {}".format(time.time()-start))

