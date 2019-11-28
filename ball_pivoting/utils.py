import os
import numpy as np
import pcl
from mesher import *


def save_obj(mesh, filename):
    print("Saving mesh to .obj file ...")
    with open(filename, 'w') as file:
        for v in mesh.vertices:
            file.write("v {0} {1} {2}\n"\
                       .format(v.xyz[0], v.xyz[1], v.xyz[2]))
        for f in mesh.facets:
            file.write("f {0} {1} {2} \n"\
                       .format(f.vertices[0].idx+1, f.vertices[1].idx+1, f.vertices[2].idx+1))
    print("Done!")


def read_vertices_from_obj(filename):
    print("Reading vertices from .obj file ...")
    points = []
    with open(filename, 'r') as file:
        for line in file:
            items = line.split(' ')
            if items[0] == 'v':
                points.append([float(items[1]), float(items[2]), float(items[3])])
    return np.array(points, dtype=np.float32)
