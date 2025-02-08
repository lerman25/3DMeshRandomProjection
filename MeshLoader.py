from Mesh import Mesh
from plyfile import PlyData
import numpy as np

def MeshFromPlyFile(filename, default_color=None):
    """
    Parse a .ply file to extract the mesh information.
    :param filename:       File to parse
    :param default_color:  Default color for the mesh surface (optional)
    :return:               List of vertices, list of faces, list of vertex colors
    """
    # Read .ply file:
    ply_data = PlyData.read(filename)
    
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
    
    return Mesh(vertices, faces, vertex_colors)