from VirtualCamera import VirtualCamera
import numpy as np

class Scene:
    def __init__(self,vc: VirtualCamera, ambient_light: np.array,light_direction_vector : np.array) -> None:
        self._vc = vc
        self._ambient_light = ambient_light
        self._light_direction_vector = light_direction_vector
    def get_vc(self):  
        return self._vc
    def get_ambient_light(self):
        return self._ambient_light
    def get_light_direction_vector(self):
        return self._light_direction_vector
    