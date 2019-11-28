import numpy as np
import pcl
from vertex import *
from edge import *
from facet import *

if __name__ == '__main__':
    v1 = Vertex(np.array([0,0,0]), np.array([0, 0, 1]))
    v2 = Vertex(np.array([0,0,1]), np.array([1, 0, 0]))
    v3 = Vertex(np.array([1,1,1]), np.array([0,1,0]))
    e = Edge(v1, v2)
    f = Facet(v1, v2, v3)
    print(v1.adj_edges)
    print(v2.adj_edges)
    print(v3.adj_edges)
    print(e.source)
    print(e.target)
    print(e.adj_facet1)
    print(e.adj_facet2)
    print(f.vertices)
