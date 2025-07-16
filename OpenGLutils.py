from Mesh import Mesh
import numpy as np
from vispy import gloo

def pass_mesh_to_opengl(mesh : Mesh,
                        attribute_position_name='a_position', attribute_color_name='a_color'):
    """
    Pass the mesh data to OpenGL for rendering.
    :param vertices:                 Array of vertex positions
    :param faces:                    Array of vertex indices defining the faces
    :param vertex_colors:            Array of RGB color per vertex
    :param attribute_position_name:  Name of the shader attribute for the vertex positions
    :param attribute_color_name:     Name of the shader attribute for the vertex colors
    :return:                         OpenGL Buffer objects
    """
    
    # Collate vertex data (position and opt. color).
    # we need to explicitly specify the data types (float32), as well as the names
    # for the variables the vertex data and th
    
    vertices = mesh.vertices
    maxas = vertices.max(axis=0)
    factor = vertices.max(axis=0)[0] - vertices.min(axis=0)[0]
    factor = 40/factor
    vertices *=factor 

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2])
    # plt.show()
    
    faces  =mesh.faces

    vertex_colors = mesh.vertex_colors
    vertices_type = [(attribute_position_name, np.float32, 3)]
    if vertex_colors is not None:
        vertices_type += [(attribute_color_name, np.float32, 3)]
        vertex_data = np.asarray(list(zip(vertices, vertex_colors)), vertices_type)
    else:
        vertex_data = np.asarray(vertices, vertices_type)

    # Buffers
    vertex_buffer = gloo.VertexBuffer(vertex_data)
    index_buffer = gloo.IndexBuffer(faces.flatten().astype(np.uint32))
    
    return vertex_buffer, index_buffer


def compute_mv_and_mvp(model_matrix, projection_matrix,
                       camera_translation_vector, camera_rotation_matrix):
    """
    Compute the MV and MVP matrices for OpenGL.
    :param model_matrix:              4x4 model matrix
    :param projection_matrix:         4x4 projection matrix
    :param camera_translation_vector: 3x1 translation vector for the camera 
    :param camera_rotation_matrix:    3x3 rotation matrix for the camera
    :return:                          4x4 MV matrix, 4x4 MVP matrix
    """

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    # View matrix (defining the camera pose):
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[:3, 3] = np.squeeze(camera_translation_vector)
    view_matrix[:3, :3] = camera_rotation_matrix

    # Converting it to OpenGL coordinate system:
    view_matrix = yz_flip.dot(view_matrix).T

    # Model-view matrix (projecting from object space to camera space):
    mv_matrix = np.dot(model_matrix, view_matrix)

    # Model-view-projection matrix (projecting from object space to image space):
    mvp_matrix = np.dot(mv_matrix, projection_matrix)
    
    return mv_matrix, mvp_matrix