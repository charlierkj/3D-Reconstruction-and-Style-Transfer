import numpy as np
import pcl
import random
import utils
from mesher import *


if __name__ == "__main__":

    cloud = pcl.PointCloud()
    points = utils.read_vertices_from_obj('bunny.obj')

    cloud.from_array(points)

    mesh = Mesher(cloud, 0.04)
    mesh.reconstruct()
    print(mesh.facets)
    print(mesh.edges_front)
    print(mesh.edges_border)
    utils.save_obj(mesh, 'test.obj')
