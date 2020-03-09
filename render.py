import random

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import util
from environment import Environment
from sphere import Sphere

"""
TODO
- planet stuff
- lights, reflection,..
"""

vp_size = np.array([3, 2])  # y, z
res = 400*vp_size  # y, z
render_count = 1
ray_distortion = 0#0.05
env = Environment(
    [
        *[Sphere(np.array([100, random.randint(-120, 120), random.randint(-120, 120)]), random.randint(1, 10) / 15, np.array([1, 1, 1])) for k
          in range(0)],
        Sphere(np.array([3, 0, 0]), 1, np.array([0.2, 0.2, 0.8])),
        Sphere(np.array([3, -3, 0]), 1, np.array([0.8, 0.2, 0.2])),
        Sphere(np.array([5, 3, 1]), 1, np.array([0.2, 0.8, 0.2])),
     ],
    [

    ]
)

mesh = util.get_viewport_mesh(vp_size, res)
render = np.zeros_like(mesh)
for i in range(render_count):
    ray_dirs = (mesh - env.eye).reshape((-1, 3))
    ray_dirs /= np.linalg.norm(ray_dirs)
    rays = np.zeros_like(ray_dirs) + env.eye
    rays += np.random.normal(0, ray_distortion, size=rays.shape)
    ray_cols = np.zeros_like(ray_dirs)

    for o in tqdm(env.objects):
        result, hit = o.intersect(rays, ray_dirs)
        result[np.invert(hit)] = rays[np.invert(hit)]
        ray_cols[hit] = o.naive_color_rays(result, ray_dirs)[hit]

        render += ray_cols.reshape((res[1], res[0], 3))
        img = hit.reshape((res[1], res[0]))
        plt.imshow(img.astype(np.float))
        plt.colorbar()
        plt.show()

        result = (result-rays).sum(axis=1)
        img = result.reshape((res[1], res[0]))
        plt.imshow(img)
        plt.colorbar()
        plt.show()
render /= render_count

plt.imshow(-render)
plt.show()

