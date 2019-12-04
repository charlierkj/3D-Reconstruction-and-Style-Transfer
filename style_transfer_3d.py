from __future__ import division
import os
import argparse
import glob
from functools import reduce

import torch
from torch import nn
from torch.nn import functional as F
import torchvision as tv
import numpy as np
from skimage.io import imread, imsave
from PIL import Image
import tqdm
import imageio

import neural_renderer as nr

class StyleTransfer3D(nn.Module):

    def __init__(self, filename_obj, filename_ref, texture_size=4, camera_distance=2.732, \
                 camera_distance_noise=0.1, elevation_min=20, elevation_max=40, \
                 lambda_style=1, lambda_content=2e9, lambda_tv=1e7, image_size=224):
        super(StyleTransfer3D, self).__init__()

        # set parameters
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.camera_distance_noise = camera_distance_noise
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.lambda_tv = lambda_tv

        # load VGG-Net
        self.vgg = tv.models.vgg16(pretrained=True).features
        for l in range(len(self.vgg)):
            if isinstance(self.vgg[l], nn.ReLU):
                self.vgg[l] = nn.ReLU(inplace=False)
            elif isinstance(self.vgg[l], nn.MaxPool2d):
                self.vgg[l] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        for param in self.vgg.parameters():
            param.requires_grad = False

        # load .obj
        vertices, faces = nr.load_obj(filename_obj)
        self.vertices = nn.Parameter(vertices[None, :, :])
        self.register_buffer('vertices_original', vertices[None, :, :])
        self.register_buffer('faces', faces[None, :, :])

        # create texture
        textures = torch.zeros(1, self.faces.shape[1], texture_size, texture_size, \
                               texture_size, 3, dtype=torch.float32)
        self.textures = nn.Parameter(textures)

        # load reference image
        transform = tv.transforms.Compose([
            tv.transforms.Resize((self.image_size, self.image_size)),
            tv.transforms.ToTensor()])
        image_ref = transform(Image.open(filename_ref).convert('RGB'))[None, ::]
        self.register_buffer('image_ref', image_ref)
        with torch.no_grad():
            features_ref = [f for f in self.vgg_features(image_ref)]
        self.features_ref = features_ref
        self.background = image_ref.mean(dim=(0,2,3))

        # setup renderer
        renderer = nr.Renderer(camera_mode='look_at', image_size=self.image_size)
        self.renderer = renderer

    def __device(self):
        return next(self.parameters()).device

    def forward(self, batch_size):
        # set random viewpoints
        self.renderer.eye = nr.get_points_from_angles(
            distance=(
                torch.ones(batch_size, dtype=torch.float32) * self.camera_distance + \
                torch.randn(batch_size).type(torch.float32) * self.camera_distance_noise),
            elevation = torch.FloatTensor(batch_size).uniform_(self.elevation_min, self.elevation_max),
            azimuth = torch.FloatTensor(batch_size).uniform_(0, 360))

        # set random lighting direction
        angles = torch.FloatTensor(batch_size).uniform_(0, 360)
        pi = torch.asin(torch.tensor(1.)) * 2
        x = torch.ones(batch_size, dtype=torch.float32) * torch.sqrt(torch.tensor(1/3))
        y = torch.ones(batch_size, dtype=torch.float32) * (1/2) * torch.sin(angles * pi /180)
        z = torch.ones(batch_size, dtype=torch.float32) * (1/2) * torch.cos(angles * pi /180)
        directions = torch.cat((x[:, None], y[:, None], z[:, None]), axis=1).to(self.__device())
        self.renderer.light_direction = directions

        # compute loss
        images, _, _ = self.renderer(self.vertices, self.faces, torch.sigmoid(self.textures))
        masks = self.renderer(self.vertices, self.faces, mode='silhouettes')
        features = self.vgg_features(images, masks)
        for i in range(len(self.features_ref)):
            self.features_ref[i] = self.features_ref[i].to(self.__device())

        loss_style = self.style_loss(features)
        loss_content = self.content_loss()
        loss_tv = self.tv_loss(images, masks)
        loss = self.lambda_style * loss_style + self.lambda_content * loss_content + self.lambda_tv * loss_tv

        # set default lighting direction
        self.renderer.light_direction = [0, 1, 0]
        
        return loss
            
    def vgg_features(self, images, masks=None, extractors=[3, 8, 15, 22]):
        mean = torch.FloatTensor([123.68, 116.779, 103.939]).to(images.device)
        h = images * 255 - mean[None, :, None, None]
        features = []
        for i, layer in enumerate(self.vgg.children()):
            h = layer(h)
            if i in extractors:
                features.append(h)

        if masks is None:
            masks = torch.ones(images.shape[0], images.shape[2], images.shape[3])

        style_features = []
        for feature in features:
            scale = masks.shape[-1] // feature.shape[-1]
            m = F.avg_pool2d(masks[:, None, :, :], kernel_size=scale, stride=scale)
            dim = feature.shape[1]

            m = m.reshape(m.shape[0], -1)
            f2 = feature.permute(0, 2, 3, 1)
            f2 = f2.reshape(f2.shape[0], -1, f2.shape[-1])
            f2 = f2 * m[:, :, None]
            f2 = torch.matmul(f2.permute(0,2,1), f2)
            f2 = f2 / (dim * m.sum(axis=1)[:, None, None])
            style_features.append(f2)

        return style_features

    def style_loss(self, features):
        loss = [torch.sum((f-fr)**2) for f, fr in zip(features, self.features_ref)]
        loss = reduce(lambda a, b : a + b, loss)
        batch_size = features[0].shape[0]
        loss /= batch_size
        return loss

    def content_loss(self):
        loss = torch.sum((self.vertices - self.vertices_original)**2)
        return loss

    def tv_loss(self, images, masks):
        s1 = (images[:, :, 1:, :-1] - images[:, :, :-1, :-1])**2
        s2 = (images[:, :, :-1, 1:] - images[:, :, :-1, :-1])**2
        masks = masks[:, None, :-1, :-1]
        masks = (masks == 1)
        return torch.sum(masks * (s1 + s2))

    
