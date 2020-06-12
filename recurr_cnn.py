#define network
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

import math 
import numpy as np

class attention(nn.Module):
    def __init__(self, feature_size=512, add_scaling_factor=False, return_attention_weights= False):
        super(attention, self).__init__()
        self.att_len = nn.Linear(feature_size, 1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.add_scaling_factor = add_scaling_factor
        self.return_attention_weights = return_attention_weights
        self.atten = []

    def forward(self, patches):
        #patch-> (N, num_patches, feature_size)
        atten_weights = self.att_len(patches).squeeze(2)#(N,num_patches)
        softmax_weights = self.softmax(atten_weights)#(N,num_patches)
        if (self.add_scaling_factor):
            softmax_weights /= np.sqrt(512) 
        #self.atten = np.append(self.atten, torch.argmax(softmax_weights, 1).cpu().data.numpy())
        patch_attention_encoding = (patches * softmax_weights.unsqueeze(2)).sum(1)#(N, feature_size)
        if (self.return_attention_weights):
            return patch_attention_encoding, softmax_weights
        return patch_attention_encoding

class rc(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes=200, num_patches=1, checkpoint=None, add_on=True):
        super(rc, self).__init__()
        self.img_features = nn.Sequential(*(list(models.vgg19(num_classes = 1000, pretrained = True).features.children())))
        self.patch_features = nn.Sequential(*(list(models.vgg19(num_classes = 1000, pretrained = True).features.children())[:-1]))
        self.add_on = nn.Sequential (
            #nn.MaxPool2d(2,2),
            nn.Conv2d(512, 1024, 1),
            #nn.BatchNorm2d(1024),
            nn.ReLU(True),  
            nn.Conv2d(1024, 1024, 1),
            #nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        if (add_on):
            self.img_classifier = nn.Linear(1024,200)
        else:
            self.img_classifier = nn.Linear(512,200)
        #self.patch_classifer =  nn.Linear(2048,200)
        self.patch_gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.img_gap = nn.AdaptiveAvgPool2d(output_size=1)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection
        self.num_patches = num_patches

        if (not checkpoint):
            self.init_weights()
        self.attention = attention(512, return_attention_weights=False)
        self.add_on_flag = add_on

    def init_weights(self):
        nn.init.normal_(self.img_classifier.weight, 0, 0.01)
        nn.init.normal_(self.fc.weight, 0, 0.01)
        nn.init.constant_(self.img_classifier.bias, 0)
        nn.init.constant_(self.fc.bias, 0)
        for child in self.add_on.children():
            if isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight,  mode='fan_out', nonlinearity='relu')
                if child.bias is not None:
                    nn.init.constant_(child.bias, 0)
            elif isinstance(child, nn.BatchNorm2d):
                nn.init.constant_(child.weight,1)
                nn.init.constant_(child.bias, 0)

    def forward(self, imgs, patches):
        #Local-Stream
        #print("Inside Model: img {} patches {}".format(imgs.shape, patches.shape))
        patches = patches.view(-1, patches.shape[-3], patches.shape[-2], patches.shape[-1])
        x = self.patch_features(patches) # N,512, 14, 14
        x = self.patch_gap(x)#(N,512)
        x = x.view(-1, self.input_size)

        # Forward propagate LSTM
        #x = x.view(-1, self.num_patches, self.input_size)
        x = x.unsqueeze(1).expand(imgs.shape[0], self.num_patches, self.input_size)
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda() # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0)) #out: tensor of shape (batch_size, seq_length, hidden_size*2)
        #print(out.shape)
        # Decode the hidden state of the last time step
        out = self.attention(out) #(N, feature_size)
        out = self.fc(out) #(N, 200)

        #GLobal-Stream
        img_features = self.img_features(imgs)
        if (self.add_on_flag):
            img_features = self.add_on(img_features)
        img_f = self.img_gap(img_features)
        img_f = img_f.view(img_f.shape[0],-1)
        return self.img_classifier(img_f), out

# class rc_1(nn.Module):

#     def __init__(self, input_size, hidden_size, num_layers, num_classes=200, num_patches=1, checkpoint=None, add_on=True):
#         super(rc_1, self).__init__()
#         self.img_features = nn.Sequential(*(list(models.vgg19(num_classes = 1000, pretrained = True).features.children())))
#         self.patch_features = nn.Sequential(*(list(models.vgg19(num_classes = 1000, pretrained = True).features.children())))
#         self.add_on = nn.Sequential (
#             #nn.MaxPool2d(2,2),
#             nn.Conv2d(512, 512, 1),
#             #nn.BatchNorm2d(1024),
#             nn.ReLU(True),  
#             nn.Conv2d(512, 512, 1),
#             #nn.BatchNorm2d(1024),
#             nn.ReLU(True)
#         )
#         if (add_on):
#             self.img_classifier = nn.Linear(1024,200)
#         else:
#             self.img_classifier = nn.Linear(512,200)
#         #self.patch_classifer =  nn.Linear(2048,200)
#         self.patch_gap = nn.AdaptiveAvgPool2d(output_size=1)
#         self.img_gap = nn.AdaptiveAvgPool2d(output_size=1)

#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
#         self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection
#         self.num_patches = num_patches

#         if (not checkpoint):
#             self.init_weights()
#         self.attention = attention(512, return_attention_weights=False)
#         self.add_on_flag = add_on

#     def init_weights(self):
#         nn.init.normal_(self.img_classifier.weight, 0, 0.01)
#         nn.init.normal_(self.fc.weight, 0, 0.01)
#         nn.init.constant_(self.img_classifier.bias, 0)
#         nn.init.constant_(self.fc.bias, 0)
#         for child in self.add_on.children():
#             if isinstance(child, nn.Conv2d):
#                 nn.init.kaiming_normal_(child.weight,  mode='fan_out', nonlinearity='relu')
#                 if child.bias is not None:
#                     nn.init.constant_(child.bias, 0)
#             elif isinstance(child, nn.BatchNorm2d):
#                 nn.init.constant_(child.weight,1)
#                 nn.init.constant_(child.bias, 0)

#     def forward(self, imgs, patches):
#         #Local-Stream
#         #print("Inside Model: img {} patches {}".format(imgs.shape, patches.shape))
#         patches = patches.view(-1, patches.shape[-3], patches.shape[-2], patches.shape[-1])
#         x = self.patch_features(patches) # N,512, 14, 14
#         x = self.add_on(x)
#         x = self.patch_gap(x)#(N,512)
#         x = x.view(-1, self.input_size)

#         # Forward propagate LSTM
#         #x = x.view(-1, self.num_patches, self.input_size)
#         x = x.unsqueeze(1).expand(imgs.shape[0], self.num_patches, self.input_size)
#         # Set initial states
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda() # 2 for bidirection
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
#         self.lstm.flatten_parameters()
#         out, _ = self.lstm(x, (h0, c0)) #out: tensor of shape (batch_size, seq_length, hidden_size*2)
#         #print(out.shape)
#         # Decode the hidden state of the last time step
#         out = self.attention(out) #(N, feature_size)
#         out = self.fc(out) #(N, 200)

#         #GLobal-Stream
#         img_features = self.img_features(imgs)
#         if (self.add_on_flag):
#             img_features = self.add_on(img_features)
#         img_f = self.img_gap(img_features)
#         img_f = img_f.view(img_f.shape[0],-1)
#         return self.img_classifier(img_f), out