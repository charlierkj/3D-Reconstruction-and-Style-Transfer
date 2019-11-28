import os
import argparse
import numpy as np
import pcl
from mesher import *
from utils import *


if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_file', type=str, \
                        default=os.path.join(current_dir, 'bunny.obj'))
    parser.add_argument('-of', '--output_file', type=str, \
                        default=os.path.join(current_dir, 'test.obj'))
    parser.add_argument('-r', '--ball_radius', default=0.04)


    cloud = pcl.PointCloud()
    print("Reading point cloud")
    if '.txt' in args.input_file:
        points = np.loadtxt(args.input_file)
    elif '.obj' in args.input_file:
        points = utils.read_vertices_from_obj(args.input_file)

    cloud.from_array(points)

    mesh = Mesher(cloud, args.ball_radius)
    mesh.reconstruct_with_multi_radius()
    utils.save_obj(mesh, args.output_file)
