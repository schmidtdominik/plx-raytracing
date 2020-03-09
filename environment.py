import numpy as np

class Environment:
    def __init__(self, objects, lights):
        self.objects = objects
        self.lights = lights
        self.eye = np.array([-1.3, 0, 0])
        # viewport is fixed at (0, 0, 0) looking into direction +x