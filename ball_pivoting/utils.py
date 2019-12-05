import os
import numpy as np
import pcl
from mesher import *
import json


def save_obj(mesh, filename):
    print("Saving mesh to .obj file ...")
    with open(filename, 'w') as file:
        file.write("# OBJ file format with ext .obj\n")
        file.write("# vertex count = {0}\n".format(mesh.n_vertices))
        file.write("# face count = {0}\n".format(mesh.n_facets))
        for v in mesh.vertices:
            file.write("v {0} {1} {2}\n"\
                       .format(v.xyz[0], v.xyz[1], v.xyz[2]))
        for f in mesh.facets:
            file.write("f {0} {1} {2} \n"\
                       .format(f.vertices[0].idx+1, f.vertices[1].idx+1, f.vertices[2].idx+1))
    print("Done!")


def read_vertices_from_obj(filename, savefile=None):
    print("Reading vertices from .obj file ...")
    points = []
    with open(filename, 'r') as file:
        for line in file:
            items = line.split(' ')
            if items[0] == 'v':
                points.append([float(items[1]), float(items[2]), float(items[3])])
    if not savefile is None:
        if '.obj' in savefile:
            with open(savefile, 'w') as file:
                for p in points:
                    file.write("v {0} {1} {2}".format(p[0], p[1], p[2]))
        elif '.txt' in savefile:
            np.savetxt(savefile, points)
    return np.array(points, dtype=np.float32)


def read_vertices_from_json(filename):
    print("Reading vertices from .json file ...")
    with open(filename, 'r') as f:
        temp = json.loads(f.read())
        point=temp['Points']
    return np.array(point, dtype=np.float32)
