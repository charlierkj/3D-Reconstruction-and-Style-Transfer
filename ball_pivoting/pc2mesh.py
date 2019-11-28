import os
import argparse
import numpy as np
import pcl
from mesher import *
import utils


if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('-if', '--input_file', type=str, \
                        default=os.path.join(current_dir, 'bunny.obj'))
    parser.add_argument('-of', '--output_file', type=str, \
                        default=os.path.join(current_dir, 'test.obj'))
    parser.add_argument('-r', '--ball_radius', nargs='*', default=0.04)
    args = parser.parse_args()


    cloud = pcl.PointCloud()
    print("Reading point cloud ...")
    if '.txt' in args.input_file:
        points = np.loadtxt(args.input_file)
    elif '.obj' in args.input_file:
        points = utils.read_vertices_from_obj(args.input_file)
    print("Reading done.")

    cloud.from_array(points)

    if isinstance(args.ball_radius, list):
        radius = [float(r) for r in args.ball_radius]
        mesh = Mesher(cloud, radius)
        mesh.reconstruct_with_multi_radius()
    else:
        radius = float(args.ball_radius)
        mesh = Mesher(cloud, radius)
        mesh.reconstruct()
        
    utils.save_obj(mesh, args.output_file)
