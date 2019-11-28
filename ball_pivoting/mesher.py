import numpy as np
import pcl
from vertex import *
from edge import *
from facet import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Mesher(object):

    def __init__(self, cloud, r):
        octree = cloud.make_octreeSearch(0.2)
        octree.add_points_from_input_cloud()
        self.octree = octree
        #self.seed
        self.vertices = []
        self.orphans = []
        ne = cloud.make_NormalEstimation()
        ne.set_KSearch(20)
        normals = ne.compute()
        for i in range(cloud.size):
            v = Vertex(np.array(cloud[i]), np.array(normals[i][0:3]), i)
            self.vertices.append(v)
            self.orphans.append(v.type == 0)
        self.n_vertics = len(self.vertices)
        self.edges_front = []
        self.edges_border = []
        self.facets = []
        self.n_facets = 0
        self.ball_radius = r
        self.sq_ball_radius = self.ball_radius**2

    def set_ball_radius(self, r):
        self.ball_radius = r

    def get_ball_radius(self, r):
        return self.ball_radius                              

    def reconstruct(self):
        print("******** Ball radius %.5f ********" % self.ball_radius)
        if len(self.edges_front) == 0:
            print("Start seeking for seed ...")
            seed = self.find_seed_triangle()
            if seed is None:
                print("No seed triangle found, triangulation done!")
            else:
                print("Seed triangle found.")
                self.add_facet(seed)
                self.expand_triangulation()
        else:
            self.expand_triangulation()

    def change_radius(self, r):
        self.ball_radius = r
        self.sq_ball_radius = self.ball_radius**2

    def update_orphans(self):
        self.orphans = [(v.type == 0) for v in self.vertices]
        
    def find_seed_triangle(self):
        idx_orphans = np.where(self.orphans)[0]
        p = self.vertices[idx_orphans[0]]
        search_point = tuple(p.xyz)
        radius = 2 * self.ball_radius
        [ind, sqdist] = self.octree.radius_search(search_point, radius)

        if len(ind) < 3:
            return None
            
        else:
            ind = ind[np.argsort(sqdist)]
            for i in range(len(ind)-1):
                q = self.vertices[i]
                for j in range(i+1, len(ind)):
                    s = self.vertices[j]
                    if not p.compatible_with(q, s):
                        continue
                    bc = self.compute_ball_center(p, q, s)
                    if bc is None:
                        continue
                    if self.empty_ball_config(p, q, s, ind, bc):
                        facet = Facet(p, q, s, bc)
                        self.edges_front.append(facet.vertices[0].get_links(facet.vertices[1]))
                        self.edges_front.append(facet.vertices[1].get_links(facet.vertices[2]))
                        self.edges_front.append(facet.vertices[2].get_links(facet.vertices[0]))
                        return facet
            return None
    

    def expand_triangulation(self):
        print("Expanding triangulation ...")
        while len(self.edges_front) > 0:
            e = self.edges_front.pop()
            es = e.source
            et = e.target
            if e.type != 1:
                continue
            
            candidate, candidate_ball_center = self.find_candidate(e)
            if (candidate is None) or (candidate.type == 2) or \
               (not candidate.compatible_with(es, et)):
                e.type = 0
                self.edges_border.append(e)
                continue

            e1 = candidate.get_links(es)
            e2 = candidate.get_links(et)
            
            if ((e1 is not None) and (e1.type != 1)) \
               or ((e2 is not None) and (e2.type != 1)):
                e.type = 0
                self.edges_border.append(e)
                continue

            facet = Facet(es, et, candidate, candidate_ball_center)
            self.add_facet(facet)

            e1 = candidate.get_links(es)
            e2 = candidate.get_links(et)

            if e1.type == 1:
                self.edges_front.append(e1)

            if e2.type == 1:
                self.edges_front.append(e2)

            if self.n_facets % 50 == 0:
                print(self.n_facets, " facets. ", len(self.edges_front), \
                      " front edges. ", len(self.edges_border), " border edges.")
        self.update_orphans()
        print("Triangulation done!")
                

    def find_candidate(self, e):
        candidate = None
        candidate_ball_center = None
        es = e.source
        et = e.target
        mp = (es.xyz + et.xyz) / 2
        opp = e.get_opposite_vertex()
        bc = e.adj_facet1.ball_center
        bc_mp = bc - mp
        r_p = np.linalg.norm(bc_mp) + self.ball_radius
        theta_min = 2 * np.pi
        [ind, sqdist] = self.octree.radius_search(tuple(mp), r_p)
        for i in ind:
            v = self.vertices[i]
            if (v is opp) or (v is es) or (v is et):
                continue
            if not v.compatible_with(es, et):
                continue
            bc_new = self.compute_ball_center(es, et, v)
            if bc_new is None:
                continue
            a = bc_mp / np.linalg.norm(bc_mp)
            b = bc_new - mp
            b = b / np.linalg.norm(b)
            theta = np.dot(a, b)
            v_d = et.xyz - es.xyz
            v_d = v_d / np.linalg.norm(v_d)
            c = np.cross(a, b)
            if np.dot(c, v_d) < 0:
                theta = 2 * np.pi - theta
            if theta > theta_min:
                continue
            if not self.empty_ball_config(es, et, v, ind, bc_new):
                continue
            theta_min = theta
            candidate = v
            candidate_ball_center = bc_new
        return candidate, candidate_ball_center


    def empty_ball_config(self, v0, v1, v2, ind, bc):
        for i in ind:
            v = self.vertices[i]
            if (v is v0) or (v is v1) or (v is v2):
                continue
            if v.distance_to(bc) < self.ball_radius:
                return False
        return True
        

    def compute_ball_center(self, v0, v1, v2):
        c = v0.distance_to(v1)
        a = v1.distance_to(v2)
        b = v2.distance_to(v0)
        p = np.array([a**2 * (b**2 + c**2 - a**2), b**2 * (a**2 + c**2 - b**2), \
                      c**2 * (a**2 + b**2 - c**2)])
        
        if np.sum(p) < 1e-30:
            return None
        p = p / np.sum(p)
        h = p[0] * v0.xyz + p[1] * v1.xyz + p[2] * v2.xyz # circumcenter of triangle
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(v0.xyz[0], v0.xyz[1], v0.xyz[2], c='g')
        #ax.scatter(v1.xyz[0], v1.xyz[1], v1.xyz[2], c='g')
        #ax.scatter(v2.xyz[0], v2.xyz[1], v2.xyz[2], c='g')
        #ax.scatter(h[0], h[1], h[2], c='r')
        #plt.show()
        rc_sq = (a**2 * b**2 * c**2) / ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
        height_sq = self.sq_ball_radius - rc_sq
        if(height_sq >= 0):
            n = self.compute_normal(v0, v1, v2)
            height = np.sqrt(height_sq)
            center = h + height * n
            return center
        return None

    def compute_normal(self, v0, v1, v2):
        n = np.cross(v1.xyz - v0.xyz, v2.xyz - v0.xyz)
        n = n / np.linalg.norm(n)
        mn = v0.xyz + v1.xyz + v2.xyz
        mn = mn / np.linalg.norm(mn)
        if np.dot(n, mn) < 0:
            n = -n
        return n

    def add_facet(self, f):
        self.facets.append(f)
        self.n_facets += 1
        
