from __future__ import division
import os
import argparse
import glob

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
from style_transfer_3d import *

def make_gif(filename):
    with imageio.get_writer(filename, mode='I') as writer:
        for filename in sorted(glob.glob('/tmp/_tmp_*.png')):
            writer.append_data(imageio.imread(filename))
            os.remove(filename)
    writer.close()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str, default=os.path.join(data_dir, 'teapot.obj'))
    parser.add_argument('-ir', '--filename_ref', type=str, default=os.path.join(data_dir, 'example3_ref.png'))
    parser.add_argument('-or', '--filename_output', type=str, default=os.path.join(data_dir, 'example3_result.gif'))
    parser.add_argument('-ls', '--lambda_style', type=float, default=1.)
    parser.add_argument('-lc', '--lambda_content', type=float, default=2e9)
    parser.add_argument('-ltv', '--lambda_tv', type=float, default=1e7)
    parser.add_argument('-emax', '--elevation_max', type=float, default=40.)
    parser.add_argument('-emin', '--elevation_min', type=float, default=20.)
    parser.add_argument('-lrv', '--lr_vertices', type=float, default=2.5e-4)
    parser.add_argument('-lrt', '--lr_textures', type=float, default=5e-2)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.732)
    parser.add_argument('-cdn', '--camera_distance_noise', type=float, default=0.1)
    parser.add_argument('-ts', '--texture_size', type=int, default=4)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-ni', '--num_iteration', type=int, default=1000)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    dir_output = os.path.dirname(args.filename_output)
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    model = StyleTransfer3D(args.filename_obj, args.filename_ref, \
                            texture_size=args.texture_size, \
                            camera_distance=args.camera_distance, \
                            camera_distance_noise=args.camera_distance_noise, \
                            elevation_min=args.elevation_min, \
                            elevation_max=args.elevation_max, \
                            lambda_style=args.lambda_style, \
                            lambda_content=args.lambda_content, \
                            lambda_tv=args.lambda_tv)
    model.cuda()

    lr_vertices = args.lr_vertices
    lr_textures = args.lr_textures
    beta1 = args.adam_beta1
    beta2 = args.adam_beta2

    optimizer_v = torch.optim.Adam([model.vertices], lr=lr_vertices, betas=(beta1, beta2))
    optimizer_t = torch.optim.Adam([model.textures], lr=lr_textures, betas=(beta1, beta2))

    # optimizing
    loop = tqdm.tqdm(range(300))
    for _ in loop:
        loop.set_description('Optimizing')
        optimizer_v.zero_grad()
        optimizer_t.zero_grad()
        loss = model(args.batch_size)
        loss.backward()
        optimizer_v.step()
        optimizer_t.step()

    # drawing
    loop = tqdm.tqdm(range(0, 360, 4))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        model.renderer.eye = nr.get_points_from_angles(2.732, 30, azimuth)
        images, _, _ = model.render(model.vertices, model.faces, torch.tanh(model.textures))
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        imsave('/tmp/_tmp_%04d.png' % num, image)
    make_gif(args.filename_output)  
    
