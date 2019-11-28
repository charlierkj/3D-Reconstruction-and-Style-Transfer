import numpy as np
import pcl
from edge import *


class Facet(object):

    def __init__(self, v0, v1, v2, ball_center=None):
        self.vertices = [v0, v1, v2]
        e0, e1, e2 = v0.get_links(v1), v1.get_links(v2), v2.get_links(v0)
        if e0 is None:
            e0 = Edge(v0, v1)
        if e1 is None:
            e1 = Edge(v1, v2)
        if e2 is None:
            e2 = Edge(v2, v0)
        e0.add_adjacent_facet(self)
        e1.add_adjacent_facet(self)
        e2.add_adjacent_facet(self)
        for i in range(3):
            self.vertices[i].add_adjacent_facet(self)
            self.vertices[i].update_type()
        self.ball_center = ball_center
        

    def set_ball_center(self, bc):
        self.ball_center = bc
        

    def get_ball_center(self):
        return self.ball_center
    

    def has_vertex(self, v):
        if v in self.vertices:
            return True
        else:
            return False

       
