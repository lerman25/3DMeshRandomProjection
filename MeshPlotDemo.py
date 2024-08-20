import numpy as np
from VirtualCamera import VirtualCamera,create_camera_matrix
from FootballPitchMeshLoader import PitchMeshLoader
from MeshOpenGL import MeshOpenGL
# PLY_FILE = 'SoccerFieldEX.ply'
PLY_FILE = 'c:/SoccerBaked.ply'
# TEXTURE = 'SoccerBakeDiffuse.jpeg'
# PLY_FILE =("C:/Users/cs-lab/Desktop/duck/duck/mesh.ply")

# PLY_FILE = 'c:/SoccerFieldEXOBJ.obj'
mesh_loader = PitchMeshLoader(PLY_FILE)
# mesh_from_ply_vtk(PLY_FILE)
# view_mesh(mesh_loader._mesh[0],mesh_loader._mesh[1])
# Camera
fx, fy = 572.4114, 573.5704   # Focal lengths
# fx = fy = 400
cx, cy = 325.2611, 242.0489   # Central point
image_size = (640, 480)       # Image size

K = create_camera_matrix(fx, fy, cx, cy)
vc = VirtualCamera(K,background_color= (0,0,0))

meshGL = MeshOpenGL(mesh_loader,vc)
meshGL.generate_random_images(10,radius=20)
meshGL.plot_generated_images()
