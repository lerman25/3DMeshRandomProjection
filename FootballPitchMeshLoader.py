# based on https://github.com/wikibook/dl-vision/blob/master/Chapter07/ch7_nb3_render_images_from_3d_models.ipynb
import os
import numpy as np
from plyfile import PlyData

def mesh_from_ply(filename, default_color=None):
    """
    Parse a .ply file to extract the mesh information.
    :param filename:       File to parse
    :param default_color:  Default color for the mesh surface (optional)
    :return:               List of vertices, list of faces, list of vertex colors
    """
    # Read .ply file:
    ply_data = PlyData.read(filename)
    # Get list of faces and vertices, as numpy arrays:
        # Get list of faces and vertices, as numpy arrays:
    faces = np.vstack(ply_data['face'].data['vertex_indices'])
    vertices = np.stack(
        [ply_data['vertex'].data['x'], ply_data['vertex'].data['y'], ply_data['vertex'].data['z']],
        axis=-1).astype(np.float32)
    # Check if file contains per-vertex color information:
    if 'blue' in ply_data['vertex']._property_lookup:
        # If so, extract the vertex colors as a numpy array:
        vertex_colors = np.stack(
            [ply_data['vertex'].data['blue'], 
             ply_data['vertex'].data['green'], 
             ply_data['vertex'].data['red']],
            axis=-1).astype(np.float32) / 255.
    elif default_color is not None:
        # Otherwise, use default color if provided:
        vertex_colors = np.tile(default_color, [vertices.shape[0], 1])
    else:
        vertex_colors = None
    print(vertex_colors)
    return vertices, faces, vertex_colors

class PitchMeshLoader():
    def __init__(self,path_to_3d : str,texture = None) -> None:
        if not os.path.exists(path_to_3d):
            raise ValueError("mesh file doesn't exists")
        self._3d_file_path = path_to_3d
        if path_to_3d.split('.')[-1]=='ply':
            self._mesh = mesh_from_ply(path_to_3d)
        else:
            raise ValueError("Only Ply format in currently supported")
        self._n_faces = self._mesh[1].shape
        self._n_vertices = self._mesh[0].shape
        self._texture = texture