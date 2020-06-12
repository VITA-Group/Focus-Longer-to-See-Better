import torch
import torchvision
#from torchvision models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from knockknock import slack_sender
#import cv2

import argparse
import os
import sys
import pickle
import random
import shutil
import time
#import h5py

from preprocess import mean, std
from preprocess import save_preprocessed_img2
#from models import *
from utils import save_accuracy_figure, save_loss_figure, get_acc_from_logits
#from data import create_img_patches
#from patch_only_models import vgg19_stackedlstm_patch_only
from recurr_cnn import rc

torch.cuda.seed_all()
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

parser = argparse.ArgumentParser()
#parser.add_argument('-gpuid', help="gpuIds")
parser.add_argument('-imgdir',  help="path to train img dir")
parser.add_argument('-trainpatches', default= "./Patches/train_patches.npy", help="Path to train .npy file containing patches bbox info")
parser.add_argument('-num_timesteps', default=4, type=int, help="number of LSTMs time steps")
parser.add_argument('-batch_size', default=8, type=int)
parser.add_argument('-num_workers', default=2, type= int)
parser.add_argument('-testimgdir', help="Path to test directory")
parser.add_argument('-testpatches',default= "./Patches/test_patches.npy", help="Path to path .npy file")
parser.add_argument('-checkpoint', required=False, type=str, help= "Load models")
parser.add_argument('-savedir', required=True, help="Path to save weigths and logs")
parser.add_argument('-xent_coef', default=1.0, type=float, help="Coefficient for cross-entropy" )
parser.add_argument('-start_epoch', default=0, type=int, help="")
parser.add_argument('-patch_size', type=int, default=224)
parser.add_argument('-img_size', type=int, default = 448)
parser.add_argument('-checkpoint_global_branch', required=False, help="Weights of global branch")
parser.add_argument('-checkpoint_patch_branch', required=False, help= "Weights of model pretrain on patches")
parser.add_argument('-test_freq', default=2, type=int, required=False, help="Frequency to run over test_dataset")
parser.add_argument('-lr', default=0.001, type= float, required=False)
parser.add_argument('-global_lr', default=0.001, type= float, required=False)
parser.add_argument('--lr_steps', nargs='+', type=int)


args= parser.parse_args()
if (not os.path.exists(args.savedir)):
    os.mkdir(args.savedir)
#Copy the file to save dir to save running configuration
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=args.savedir)
#shutil.copy(src=os.path.join(os.getcwd(), 'models.py'), dst=args.savedir)
print(args)
# npPatches = np.load(args.trainpatches).astype(int) #(N,10,5)
# test_npPatches = np.load(args.testpatches).astype(int)

epochs = 500
print("Running on GPU ID: {} with process id: {}".format(os.environ['CUDA_VISIBLE_DEVICES'], os.getpid()))

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, type='Train', *arg):
        super(ImageFolderWithPaths, self).__init__(*arg)
        self.type = type
        if (self.type == 'Train'):
            self.npPatches = np.load(args.trainpatches).astype(int)
        else:
            self.npPatches = np.load(args.testpatches).astype(int)

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        img_coords_for_patches = self.npPatches[index]
        assert np.all(img_coords_for_patches[:,-1]==original_tuple[1]), "{} {}".format(img_coords_for_patches[:,-1] ,original_tuple[1])
        patches = torch.zeros((1, 3, args.patch_size, args.patch_size))
        
        choices = random.choices(range(10))
        for i in range(len(choices)):
            #ith_patch -> unnormalized tensor corresponding patch
            patch_idx = choices[i]
            ith_patch = original_tuple[0][:,img_coords_for_patches[patch_idx][0]:img_coords_for_patches[patch_idx][1],
                                             img_coords_for_patches[patch_idx][2]:img_coords_for_patches[patch_idx][3]]
            #save_preprocessed_img('./testing_loader/path_{}_{}.png'.format(index, i), ith_patch)
            ith_patch = ith_patch.unsqueeze(0)
            #print("Patch  shape {}".format(ith_patch.shape))

            patches[i] = normalize(F.interpolate(ith_patch, size=(args.patch_size,args.patch_size), mode='bilinear',  align_corners=True).squeeze(0))

        return (normalize(original_tuple[0]), original_tuple[1], patches)

normalize = transforms.Normalize(mean=mean,
                                 std=std)
T = transforms.Compose([
    transforms.Resize(size=(args.img_size, args.img_size)),
    transforms.ToTensor() #DO NOT NORMALIZE HERE
    ])

train_dataset = ImageFolderWithPaths('Train', args.imgdir,  T)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
total_train_images = len(train_loader.dataset)

test_dataset = ImageFolderWithPaths('Test', args.testimgdir, T)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
total_test_images = len(test_loader.dataset)

print("Total train images: {}".format(len(train_loader.dataset)))
print("Total test images: {}".format(len(test_loader.dataset)))

net = rc(input_size=512 ,hidden_size=512, num_layers=2, num_patches=args.num_timesteps, checkpoint=args.checkpoint_patch_branch)

print(net)
#Feature-Extraction (512*14*14)-> GAP(512)-> concat patches (512*4=2048) -> classifier(2048x200)
net = nn.DataParallel(net)
net = net.cuda()
if (args.checkpoint_global_branch):
    print("Loading Pretrain weights for global branch {}".format(args.checkpoint_global_branch))
    load_weights = torch.load(args.checkpoint_global_branch)
    model_dict = net.state_dict()
    #print(model_dict.keys())
    global_params_dict = {}
    for (key, value) in load_weights['model'].items():
        if 'features' in key:
            global_params_dict[key.replace('features', 'img_features')] = value
        elif 'classifier' in key:
            global_params_dict[key.replace('classifier', 'img_classifier')] = value
        else:#add_on
            global_params_dict[key] = value
    #print(global_params_dict.keys())
    model_dict.update(global_params_dict)
    net.load_state_dict(model_dict)

if (args.checkpoint_patch_branch):
    print("Loading Pretrain weights for patch branch {}".format(args.checkpoint_patch_branch))
    checkpoint = torch.load(args.checkpoint_patch_branch)
    #del checkpoint['model']['module.patch_classifer2.weight']
    #del checkpoint['model']['module.patch_classifer2.bias']
    model_dict = net.state_dict()
    model_dict.update(checkpoint['model'])
    net.load_state_dict(model_dict)
best_acc = 0.0

def test(net, dataloader):
    with torch.no_grad():
        net.eval()
        corrects, patch_corrects, global_corrects = 0.,0.,0.
        for (imgs, labels, patches) in dataloader:
            images, patches, labels = imgs.cuda(), patches.cuda(), labels.cuda()
            
            img_logits, patch_logits = net(images, patches)
            #Attention/model networks
            weighted_logits = img_logits + args.xent_coef * (patch_logits)
            _, predicts = torch.max(weighted_logits, 1)
            corrects += torch.sum(predicts == labels).item()
    curr_acc = (corrects/total_test_images)
    print("Test: Total acc {:.5f}".format(curr_acc))
    net.train()
    return curr_acc


#losses
criteria = nn.CrossEntropyLoss()
group_params = [
    { 'params' : net.module.img_features.parameters(), 'lr': args.global_lr},
    { 'params' : net.module.add_on.parameters(), 'lr': args.global_lr },
    { 'params' : net.module.img_classifier.parameters(), 'lr': args.global_lr },
    { 'params' : net.module.lstm.parameters()},
    { 'params' : net.module.fc.parameters()},
    { 'params' : net.module.patch_features.parameters()},
    { 'params' : net.module.attention.parameters()}
]
#Optimizer
optimizer = optim.SGD(group_params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps)

if (args.checkpoint):   
    print("Loading from checkpoint....")
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['model'])
    args.start_epoch = checkpoint["start_epoch"]
    best_acc = checkpoint["best_acc"]
    optimizer.load_state_dict(checkpoint['optimizer'])

#test(net, test_loader)
#start training
def train(best_acc):
    print("Starting training....")
    curr_acc = 0.0
    loss1 = loss1_1 = loss1_2 = 0.0
    loss1_list, loss2_list = [], []
    train_acc, test_acc = [], []
    for epoch in range(args.start_epoch, epochs):
        total_loss = 0.
        scheduler.step()
        corrects = 0
        start = time.time()
        for idx, (images, labels, patches) in enumerate(train_loader):
            #save_preprocessed_img2('./testing_loader/path_{}_{}.png'.format(index, labels[epoch].item()), patches_batch, index)
            images, labels, patches = images.cuda(), labels.cuda(), patches.cuda() #patches-> (N,2,16,3,224,224)
            optimizer.zero_grad()
            img_preds, patch_preds = net(images, patches) #(N, 200) , (N, num_sub_patches, 200)

            loss1_1 = criteria(img_preds, labels)
            #Attention/model networks
            loss1_2 =  criteria(patch_preds, labels)
            #last step LSTM
            loss = loss1_1 + args.xent_coef * loss1_2

            loss.backward()
            #plot_grad_flow(net.named_parameters())
            optimizer.step()

            total_loss += loss.item()

            #Attentions/model networks
            weighted_logits = img_preds +args.xent_coef * patch_preds
            predicts = torch.argmax(weighted_logits, 1)
            corrects += torch.sum(predicts == labels).item()
            
            loss1_list.append(loss1_1.item())
            loss2_list.append(loss1_2.item())

            if (idx%500)==0:
                print("Epoch {} Iter {} Avg error till now: {:.3f} ".format(epoch, idx, total_loss/(idx+1)))
        #print("Number of corrects: {}".format(corrects.item()))
        curr_train_acc = corrects/total_train_images
        train_acc.append(curr_train_acc)

        print("Epoch {} Accuracy {:.4f} time taken {:.3f}".format(epoch, curr_train_acc, (time.time()-start)))
        
        if (epoch % args.test_freq==0):
            curr_acc = test(net, test_loader)
            test_acc.append(curr_acc)
            print("Epoch {} Testing Accuracy: {:.5f}".format(epoch, curr_acc))
            if (best_acc < curr_acc):
                print("**NEW BEST ACCURACY {:.5f} at epoch {}**".format(curr_acc, epoch))
                torch.save({ 'start_epoch': epoch+1,
                            'model': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_acc': curr_acc
                    }, args.savedir + '/checkpoint_'+ str(epoch) + "_" + str(curr_acc) +'.pth')
                best_acc = curr_acc
        # if (epoch in (0, 20,30, 35 ,40 ,45, 50, 75,100)):
        #     save_accuracy_figure(train_acc, test_acc, global_branch_acc, patch_branch_acc, args.savedir, args.test_freq)
        #     save_loss_figure(loss1_list, loss2_list, args.savedir)

train(best_acc)

