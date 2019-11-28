import numpy as np
import pcl


class Edge(object):

    def __init__(self, s, t):
        self.source = s
        self.target = t
        self.source.add_adjacent_edge(self)
        self.target.add_adjacent_edge(self)
        self.adj_facet1 = None
        self.adj_facet2 = None
        self.type = 1 # 0: border; 1: front; 2: inner
        self.len = np.linalg.norm(self.source.xyz - self.target.xyz)

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def get_facet1(self):
        return self.adj_facet1

    def get_facet2(self):
        return self.adj_facet2

    def add_adjacent_facet(self, f):
        if (f is self.adj_facet1) or (f is self.adj_facet2):
            return False
        if self.adj_facet1 is None:
            self.adj_facet1 = f
            self.set_type(1)
            return True
        if self.adj_facet2 is None:
            self.adj_facet2 = f
            self.set_type(2)
            return True
        print("Already two triangles")
        return False

    def remove_adjacent_facet(self, f):
        if self.adj_facet1 is f:
            self.adj_facet1 = None
            self.set_type(1)
            return True
        if self.adj_facet2 is f:
            self.adj_facet2 = None
            self.set_type(2)
            return True
        return False

    def has_vertex(self, v):
        if (self.source is v) or (self.target is v):
            return True
        return False
    
    def update_orientation(self):
        opp = self.get_oppposite_vertex()
        v = np.cross(self.target.xyz-self.source.xyz, opp.xyz-self.source.xyz)
        v = v / np.linalg.norm(v)
        n = self.source.xyz + self.target.xyz + opp.xyz
        n = n / np.linalg.norm(n)
        if np.dot(v, n) < 0:
            self.source, self.target = self.target, self.source

    def get_opposite_vertex(self):
        if self.adj_facet1 is None:
            return None
        for i in range(3):
            opp = self.adj_facet1.vertices[i]
            if (opp is not self.source) and (opp is not self.target):
                return opp
        return None

    def set_type(self, ty):
        self.type = ty

    def get_type(self):
        return self.type
        
            
