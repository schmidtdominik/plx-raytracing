import numpy as np

def get_viewport_mesh(vp_size, res):
    y = np.linspace(-vp_size[0] / 2, vp_size[0] / 2, res[0])
    z = np.linspace(-vp_size[1] / 2, vp_size[1] / 2, res[1])
    ys, zs = np.meshgrid(y, z)
    mesh = np.concatenate([np.zeros((res[1], res[0], 1)), np.expand_dims(ys, axis=-1), np.expand_dims(zs, axis=-1)], axis=2)
    return mesh