from PlotUtils import plot_image_grid
from ImageGenerator import ImageGenerator
from VirtualCamera import VirtualCamera as VC
from SphereAlgoirthms import fibonacci_sphere
from Canvas import Canvas
from Scene import Scene
import numpy as np
import MeshLoader
import os 
import matplotlib.pyplot as plt

def main():

    mesh_file = os.path.join('','duck','mesh.ply')
    mesh = MeshLoader.MeshFromPlyFile(mesh_file, default_color=(0.5, 0.5, 0.5))

    image_size = (640,480)

    # Create a virtual camera:
    fx, fy = 572.4114, 573.5704   # Focal lengths
    cx, cy = 325.2611, 242.0489   # Central point

    K = VC.create_camera_matrix(fx, fy, cx, cy)
    vc = VC(K, image_size=image_size)

    # Randomize lighting confitions:

    canvas = Canvas(image_size=image_size, background_color=(0, 0, 0))
    canvas.load_mesh_on_canvas(mesh)

    ambient_light = np.random.uniform(0.05, 0.40)
    ambient_light = 100
    directional_light_vector = np.random.uniform(-1, 1, size=3)
    scene = Scene(vc,ambient_light,directional_light_vector)

    ig = ImageGenerator(scene,canvas,fibonacci_sphere)

    rgb_images = ig.generate_image(10)
    figure = plot_image_grid([rgb_images], transpose=True)


if __name__ == "__main__":
    main()