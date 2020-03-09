import math
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cosine



class Planet:

    def __init__(self, pos, r, base_col):
        self.pos = pos
        self.r = r
        self.base_col = base_col

    def inside(self, rpos):
        dists = np.square(self.pos - rpos).sum(axis=1)
        return np.sqrt(dists) <= self.r

    def intersect(self, rays, ray_dirs):
        """
        # ray_dirs have to be normalized
        term1 = -((rays-self.pos) * ray_dirs).sum(axis=1) / (ray_dirs*ray_dirs).sum(axis=1)
        discriminant = (((2*((rays-self.pos) * ray_dirs).sum(axis=1)))**2 -((2*((rays-self.pos) * ray_dirs).sum(axis=1))**2 * (((rays-self.pos)*(rays-self.pos)).sum(axis=1) - self.r**2))) / (2*(ray_dirs*ray_dirs).sum(axis=1))




                ray_dirs = ray_dirs / np.abs(np.expand_dims(ray_dirs.sum(axis=1), axis=1))
        # ray_dirs have to be normalized
        oc = rays - self.pos
        a = (ray_dirs * ray_dirs).sum(axis=1)
        b = 2 * (oc * ray_dirs).sum(axis=1)
        c = (oc * oc).sum(axis=1) - self.r ** 2
        discriminant = b * b - 4 * a * c
        hit = discriminant > 0
        term1 = -b / (2*a)
        term2 = np.sqrt(discriminant) / 2*a
        sol1 = term1 + term2
        sol2 = term1 - term2
        sol1[sol1 < 0] = np.inf
        sol2[sol2 < 0] = np.inf
        sel_sols = np.minimum(sol1, sol2)
        """

        ray_dirs = ray_dirs / np.abs(np.expand_dims(ray_dirs.sum(axis=1), axis=1))
        oc = rays - self.pos
        a = (ray_dirs * ray_dirs).sum(axis=1)
        b = 2 * (oc * ray_dirs).sum(axis=1)
        c = (oc * oc).sum(axis=1) - self.r ** 2
        discriminant = b**2 - 4 * a * c
        #hit = discriminant > 0 # or >= to include tangents
        term1 = -b / (2*a)
        term2 = np.sqrt(discriminant) / (2*a)

        sol1 = term1 + term2
        sol2 = term1 - term2
        sol1[sol1 < 0] = np.inf
        sol2[sol2 < 0] = np.inf
        sel_sols = np.minimum(sol1, sol2)
        hit = np.invert(np.isnan(sel_sols))
        res = rays+ray_dirs*np.expand_dims(sel_sols, axis=1)
        res[np.invert(hit)] = np.nan
        return res, hit

    def color_rays(self, rays, ray_dirs):
        ray_dirs = ray_dirs / np.abs(np.expand_dims(ray_dirs.sum(axis=1), axis=1))
        surface_normals = rays - self.pos
        surface_normals /= np.linalg.norm(surface_normals)
        cosine_dist = (ray_dirs * surface_normals).sum(axis=1) / (np.linalg.norm(ray_dirs, axis=1) * np.linalg.norm(surface_normals, axis=1))
        result = np.expand_dims(self.base_col, 0) * np.expand_dims(cosine_dist, 1)
        return result  # np.zeros_like(ray_dirs) + self.base_col

#p = Planet(np.array([3, 0, 0]), 1, np.array([0.8, 0.2, 0.2]))
#print(p.intersect(np.array([[0, 0.2, 0.5]]), np.array([[1, 0, 0]])))
#exit()

class Environment:
    def __init__(self, objects, lights):
        self.objects = objects
        self.lights = lights
        self.eye = np.array([-1, 0, 0])

env = Environment(
    [
        Planet(np.array([3, 0, 0]), 1, np.array([0.2, 0.2, 0.8])),
        Planet(np.array([3, -3, 0]), 1, np.array([0.8, 0.2, 0.2])),
        Planet(np.array([5, 3, 1]), 1, np.array([0.2, 0.8, 0.2]))
     ],
    [

    ]
)

vp_size = np.array([2, 2])  # y, z
res = 500*vp_size  # y, z
y = np.linspace(-vp_size[0]/2, vp_size[0]/2, res[0])
z = np.linspace(-vp_size[1]/2, vp_size[1]/2, res[1])
ys, zs = np.meshgrid(y, z)
mesh = np.concatenate([np.zeros((res[1], res[0], 1)), np.expand_dims(ys, axis=-1), np.expand_dims(zs, axis=-1)], axis=2)
ray_dirs = (mesh - env.eye).reshape((-1, 3))
ray_dirs /= np.linalg.norm(ray_dirs)
rays = np.zeros_like(ray_dirs) + env.eye
ray_cols = np.zeros_like(ray_dirs)

for o in env.objects:
    result, hit = o.intersect(rays, ray_dirs)
    result[np.invert(hit)] = rays[np.invert(hit)]
    ray_cols[hit] = o.color_rays(result, ray_dirs)[hit]

    img = hit.reshape((res[1], res[0]))
    plt.imshow(img.astype(np.float))
    plt.colorbar()
    plt.show()
    img = result.reshape((res[1], res[0], 3))
    plt.imshow(img)
    plt.colorbar()
    plt.show()

#if i % 10 == 0:
img = ray_cols.reshape((res[1], res[0], 3))
print(np.max(img))
plt.imshow(-img)
plt.colorbar()
plt.show()

