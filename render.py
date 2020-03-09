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

http://graphics.cs.cmu.edu/nsp/course/15-462/Spring04/slides/13-ray.pdf
"""

vp_size = np.array([3, 2])  # y, z
res = 500*vp_size  # y, z
render_count = 1
ray_distortion = 0#0.05
env = Environment(
    [
        *[Sphere(np.array([80, random.randint(-120, 120), random.randint(-120, 120)]), random.randint(1, 10) / 15, np.array([1, 1, 1])) for k
          in range(250)],
        Sphere(np.array([20, 0, -5]), 1, np.array([1, 0, 0])),
        Sphere(np.array([3, 2, 0]), 1, np.array([0, 1, 0])),
        Sphere(np.array([4, -2, 1]), 1, np.array([0, 0, 1])),
    ],
    [
        Sphere(np.array([2, 2, 0]), 2, np.array([1, 1, 1]))
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

    render_depth_map = np.full(res[1] * res[0], np.inf)

    for o in tqdm(env.objects):
        result_w_inf, hitmap = o.intersect(rays, ray_dirs)

        next_ray_pos = np.copy(result_w_inf)
        next_ray_pos[np.invert(hitmap)] = rays[np.invert(hitmap)] # replace nans by original ray location

        dist_map_w_inf = np.abs((rays - result_w_inf)).sum(axis=1)
        #new_color_patch = o.naive_color_rays(next_ray_pos, ray_dirs)[hitmap]

        # nonrecursive first bounce

        rays_ss = next_ray_pos[hitmap]
        ray_dirs_ss = rays[hitmap]
        sphere_normals = rays_ss - o.pos
        sphere_normals /= np.linalg.norm(sphere_normals)

        refl_dir = 2*(sphere_normals * ray_dirs_ss) * sphere_normals - ray_dirs_ss

        """
        l = env.lights[0]
        _, light_hits = l.intersect(rays_ss, refl_dir)
        print(np.mean(light_hits))
        new_color_patch[light_hits] += 1#(new_color_patch[light_hits] + 1)/2"""

        """l = env.lights[0]
        rays_ss = next_ray_pos[hitmap]
        shadow_rays = l.pos - rays_ss
        _, light_hits = l.intersect(rays_ss, refl_dir)"""

        #

        new_color_patch = o.naive_color_rays(rays_ss, refl_dir)

        old_color_patch = ray_cols[hitmap]
        new_dist_map_patch = dist_map_w_inf[hitmap]
        old_dist_map_patch = render_depth_map[hitmap]
        ray_cols[hitmap] = np.where(np.expand_dims(new_dist_map_patch < old_dist_map_patch, 1), new_color_patch, old_color_patch)
        render_depth_map[hitmap] = np.minimum(new_dist_map_patch, old_dist_map_patch)
        """img = hitmap.reshape((res[1], res[0]))
        plt.imshow(img.astype(np.float))
        plt.colorbar()
        plt.show()

        plt.imshow(np.nan_to_num(dist_map, copy=True, nan=-1, posinf=-1, neginf=-1).reshape(res[1], -1))
        plt.colorbar()
        plt.show()
        
        plt.imshow(o.naive_color_rays(next_ray_pos, ray_dirs).reshape((res[1], res[0], 3)))
        plt.show()"""

    render = ray_cols.reshape((res[1], res[0], 3))
render /= render_count


plt.imshow(render_depth_map.reshape(res[1], res[0]))
plt.colorbar()
plt.show()
plt.imshow(render)
plt.show()
