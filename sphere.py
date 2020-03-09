import numpy as np


class Sphere:

    def __init__(self, pos, r, base_col):
        self.pos = pos
        self.r = r
        self.base_col = base_col

    def inside(self, rpos):
        dists = np.square(self.pos - rpos).sum(axis=1)
        return np.sqrt(dists) <= self.r

    def intersect(self, rays, ray_dirs):
        ray_dirs = ray_dirs / np.abs(np.expand_dims(ray_dirs.sum(axis=1), axis=1))
        oc = rays - self.pos
        a = (ray_dirs * ray_dirs).sum(axis=1)
        b = 2 * (oc * ray_dirs).sum(axis=1)
        c = (oc * oc).sum(axis=1) - self.r ** 2
        discriminant = b**2 - 4 * a * c
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

    def naive_color_rays(self, rays, ray_dirs):
        ray_dirs = ray_dirs / np.abs(np.expand_dims(ray_dirs.sum(axis=1), axis=1))
        surface_normals = rays - self.pos
        surface_normals /= np.linalg.norm(surface_normals)
        cosine_dist = (ray_dirs * surface_normals).sum(axis=1) / (np.linalg.norm(ray_dirs, axis=1) * np.linalg.norm(surface_normals, axis=1))
        result = np.expand_dims(self.base_col, 0) * np.expand_dims(cosine_dist, 1)
        return np.abs(result)  # np.zeros_like(ray_dirs) + self.base_col


p = Sphere(np.array([3, 0, 0]), 1, np.array([0.2, 0.2, 0.8]))
print(p.intersect(np.array([[0, 0.5, 0.5], [0, -0.5, -0.5]]), np.array([[1, 0, 0], [1, 0, 0]])))
