from VirtualCamera import VirtualCamera as VC
from Scene import Scene
from Canvas import Canvas
import numpy as np
from collections.abc import Callable


class ImageGenerator:
    def __init__(self,scene: Scene,canvas: Canvas,sphereAlg : Callable) -> None:
        self._scene = scene
        self._canvas = canvas
        self._sphereAlg = sphereAlg

    def generate_image(self,num_images: int):
        camera_positions = self._sphereAlg(radius=80, samples=num_images, randomize=False)
        camera_transform_matrices = [VC.look_at(camera_position, self._canvas.model_matrix[0:3, 3])
                           for camera_position in camera_positions]
        camera_positions = [camera_transform_matrix[0:3, 3] 
                            for camera_transform_matrix in camera_transform_matrices]
        camera_rotation_matrices = [camera_transform_matrix[0:3, 0:3]
                            for camera_transform_matrix in camera_transform_matrices]

        rgb_images = []
        for camera_position, camera_rotation_matrix in zip(camera_positions, camera_rotation_matrices):
            
            rgb_image = self._canvas.draw(self._scene,camera_position, camera_rotation_matrix)
            rgb_images.append(rgb_image)
            
        rgb_images = np.asarray(rgb_images)
        return rgb_images