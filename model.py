# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
@Modified by: Said Ohamouddou
@Date: 2024/10/26
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan_convs import KANConv2DLayer 
from kan_convs import KAGNConv1DLayer
from kans import KANLayer


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class KANDGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(KANDGCNN, self).__init__()
        self.args = args
        self.k = args.k  # Number of nearest neighbors to consider in graph construction
        
        # Define convolutional layers with customized KANConv layers for feature extraction
        self.conv1 = nn.Sequential(KANConv2DLayer(6, 128, kernel_size=1))  # Initial layer with input of 6 channels, outputting 128 features
        self.conv2 = nn.Sequential(KANConv2DLayer(64 * 2, 64, kernel_size=1))  # Commented out: second conv layer for further processing

        # Additional convolutional layers (currently commented out) for potential deeper feature extraction
        # self.conv3 = nn.Sequential(Conv2d(64 * 2, 128, kernel_size=1), self.bn3)
        # self.conv4 = nn.Sequential(Conv2d(128 * 2, 256, kernel_size=1), self.bn4)

        # Last convolutional layer using a 1D layer for embedding features
        self.conv5 = nn.Sequential(KAGNConv1DLayer(128, args.emb_dims, kernel_size=1))
        
        # Linear layers for classification or final output, currently commented out
        self.linear1 = KANLayer(args.emb_dims * 2, output_channels)
        # self.linear2 = KANLayer(512, 256)
        # self.linear3 = KANLayer(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size
        
        # Step 1: Construct graph-based features for each point
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3 * 2, num_points, k)
        
        # Step 2: Apply the first convolutional layer
        x = self.conv1(x)  # Transforms to (batch_size, 64, num_points, k)
        
        # Step 3: Max-pooling to reduce feature dimensionality
        x1 = x.max(dim=-1, keepdim=False)[0]  # Reduces along the k dimension: (batch_size, 64, num_points)
        
        # Additional convolution layers (commented out), demonstrating further processing steps if uncommented
        # x = get_graph_feature(x1, k=self.k)
        # x = self.conv2(x)
        # x2 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x2, k=self.k)
        # x = self.conv3(x)
        # x3 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x3, k=self.k)
        # x = self.conv4(x)
        # x4 = x.max(dim=-1, keepdim=False)[0]

        # Uncommenting this line would concatenate feature maps from multiple layers
        # x = torch.cat((x1, x2), dim=1)  # (batch_size, 64+64+128+256, num_points)

        # Final convolution to embed the extracted features
        x = self.conv5(x1)  # Embeds the features: (batch_size, 64, num_points) -> (batch_size, emb_dims, num_points)

        # Step 4: Global pooling for fixed-size feature vector per point cloud
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # Max pooling: (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # Avg pooling: (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        
        # Concatenate the pooled features for richer representations
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims * 2)

        # Optional linear layers (currently commented out)
        # x = self.linear1(x)  # Further feature transformation: (batch_size, emb_dims * 2) -> (batch_size, 512)
        # x = self.linear2(x)  # Reduction: (batch_size, 512) -> (batch_size, 256)
        
        # Final transformation to the output dimension
        x = self.linear1(x)  # Produces final output: (batch_size, 256) -> (batch_size, output_channels)
        
        return x
