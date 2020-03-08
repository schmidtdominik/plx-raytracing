import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cosine

"""
lights, cameras, composite objects
"""

class Sphere:

    def __init__(self, pos, r, base_col):
        self.pos = pos
        self.r = r
        self.base_col = base_col

    def inside(self, rpos):
        dists = np.square(self.pos - rpos).sum(axis=1)
        return np.sqrt(dists) <= self.r

    def color_rays(self, rays, ray_dirs):
        surface_normals = rays - self.pos
        surface_normals /= np.linalg.norm(surface_normals)
        cosine_dist = (ray_dirs * surface_normals).sum(axis=1) / (np.linalg.norm(ray_dirs, axis=1) * np.linalg.norm(surface_normals, axis=1))
        result = np.expand_dims(self.base_col, 0) * np.expand_dims(cosine_dist, 1)
        return result

class Environment:

    def __init__(self, objects):
        self.objects = objects
        self.eye = np.array([-1, 0, 0])

env = Environment([
    Sphere(np.array([3, 0, 0]), 1, np.array([0.2, 0.2, 0.8])),
    Sphere(np.array([3, -3, 0]), 1, np.array([0.8, 0.2, 0.2])),
    Sphere(np.array([5, 3, 1]), 1, np.array([0.2, 0.8, 0.2]))
])

y_res = 300
z_res = 300
y = np.linspace(-1, 1, y_res)
z = np.linspace(-1, 1, z_res)
ys, zs = np.meshgrid(y, z)
mesh = np.concatenate([np.zeros((y_res, z_res, 1)), np.expand_dims(ys, axis=-1), np.expand_dims(zs, axis=-1)], axis=2)
ray_dirs = (mesh - env.eye).reshape((-1, 3))
step_size = 0.1
ray_dirs /= np.linalg.norm(ray_dirs) * step_size
rays = np.zeros_like(ray_dirs) + env.eye
ray_cols = np.zeros_like(ray_dirs)

for i in tqdm(range(100)):
    for o in env.objects:
        insides = o.inside(rays)
        #print(o.color_rays(rays, ray_dirs)[insides])
        ray_cols[insides] = o.color_rays(rays, ray_dirs)[insides]

        rays += ray_dirs

#if i % 10 == 0:
img = ray_cols.reshape((y_res, z_res, 3))
plt.imshow(img)
plt.show()
