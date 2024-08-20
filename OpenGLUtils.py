import numpy as np
from VirtualCamera import VirtualCamera
from vispy import app, gloo
import OpenGL.GL as gl
from FootballPitchMeshLoader import PitchMeshLoader

def prepare_opengl(pmesh : PitchMeshLoader,vc : VirtualCamera):
    app.use_app('PyGlet')  # Set backend (try e.g. "PyQt5" otherwise)
    image_size  = vc._image_size
    canvas = app.Canvas.__init__(show=False, size=image_size)
    tex_shape = (image_size[1], image_size[0]) # coordinates are inverted in OpenGL
    # Texture to render the image into:
    image_texture = gloo.Texture2D(shape=tex_shape + (3,))

    # Corresponding FBO (frame buffer):
    fbo = gloo.FrameBuffer(image_texture, gloo.RenderBuffer(tex_shape))
    fbo.activate()
    gloo.set_state(depth_test=True, blend=False, cull_face=False)
    gloo.set_clear_color(vc._background_color)
    gloo.set_viewport(0, 0, *canvas.size)

class OpenGL_constant:
    VERTEX_SHADER_BASIC = """
    uniform mat4 u_mv;             // Model-View matrix
    uniform mat4 u_mvp;            // Model-View-Projection matrix
    uniform vec3 u_light_position; // Position of the directional light source

    attribute vec3 a_position;     // Vertex position
    attribute vec3 a_color;        // Vertex color

    varying vec4 v_color;          // RGBA vertex color (to be passed to fragment shader)
    varying vec3 v_eye_position;   // Vertex position in eye/camera coordinates
    varying vec3 v_light;          // Vector from vertex to light source

    void main() {
        // Projected position:
        gl_Position = u_mvp * vec4(a_position, 1.0);
        // Vertex color (varying):
        v_color = vec4(a_color, 1.0);
        // Vertex position in eye/camera coordinates:
        v_eye_position = (u_mv * vec4(a_position, 1.0)).xyz;
        // Vector to the light:
        v_light = normalize(u_light_position - v_eye_position);
    }
    """

    FRAGMENT_SHADER_COLOR = """
    uniform float u_light_ambient; // Intensity of the ambient light
    varying vec4 v_color;          // Interplated vertex color
    varying vec3 v_eye_position;   // Interplated vertex position in eye/camera coordinates
    varying vec3 v_light;          // Interplated vector from vertex to light source

    void main() {
        // Face normal in eye coordinates:
        vec3 face_normal = normalize(cross(dFdx(v_eye_position), dFdy(v_eye_position)));
        // Light received by the surface (ambient + diffuse):
        float light_diffuse_w = max(dot(normalize(v_light), normalize(face_normal)), 0.0);
        float light_w = u_light_ambient + light_diffuse_w;
        light_w = clamp(light_w, 0.0, 1.0); // Clamp/clip brightness factor
        gl_FragColor = light_w * v_color;
    }
    """
    def __init__(self):
    # Set constants from separate classes as attributes
        for key, value in OpenGL_constant.__dict__.items():
            if not key.startswith("__"):
                self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")
def pass_mesh_to_opengl(mesh_loader : PitchMeshLoader,
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
    
    vertices = mesh_loader._mesh[0]
    maxas = vertices.max(axis=0)
    factor = vertices.max(axis=0)[0] - vertices.min(axis=0)[0]
    factor = 40/factor
    vertices *=factor 

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(vertices[:,0],vertices[:,1],vertices[:,2])
    # plt.show()
    
    faces  =mesh_loader._mesh[1]

    vertex_colors = mesh_loader._mesh[2]

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

import scipy.linalg
import random
def fibonacci_sphere(radius=100., samples=1, randomize=True,yeild_above = False,above_axis=1):
    """ Yields 3D cartesian coordinates of pseudo-equidistributed points on the surface of a sphere of given radius,
    aligned on the origin, using Fibonacci Sphere algorithm.
    Gist from Snord (http://stackoverflow.com/a/26127012/624547)
    @yield 3D point
    """
    rnd = 1.
    if randomize:
        rnd = random() * samples

    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r
        vals = [x,y,z]
        ret_val = [radius * x, radius * y, radius * z]
        if yeild_above:
            ret_val[above_axis] = radius*(max(-1/np.sqrt(2),vals[above_axis]))
        else:
            yield ret_val
def random_point_on_upper_hemisphere(radius=20,samples = 1):
    # Convert degrees to radians
    min_angle = np.radians(105)  # Lower limit (closer to equator, 15 degrees above x-y plane)
    max_angle = np.radians(135)  # Upper limit (45 degrees above x-y plane)

    for i in range(samples):
        # Generate a random angle phi in the range [0, 2*pi) for the azimuthal angle
        phi = np.random.uniform(np.radians(210),np.radians(180+150))
        
        # Generate a random angle theta within the specified range
        theta = np.random.uniform(min_angle, max_angle)
        
        # Convert spherical coordinates to Cartesian coordinates
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)  # Negative z to match your correction
    
        yield [x, y, z]
        
def look_at(camera_position, target_position, roll_angle=0):
    """
    Return the rotation matrix so that the camera faces the target.
    Snippet by Wadim Kehl (https://github.com/wadimkehl/ssd-6d/blob/master/rendering)
    :param camera_position:     Camera position/translation
    :param target_position:     Target position
    :param roll_angle:          Roll angle (in degrees)
    :return:                    4x4 transformation matrix
    """
    eye_direction = target_position - camera_position
    # Compute what is the "up" vector of the camera:
    if eye_direction[0] == 0 and eye_direction[1] == 0 and eye_direction[2] != 0:
        up = [-1, 0, 0]
    else:
        up = [0, 0, 1]

    # Compute rotation matrix:
    rotation_matrix = np.zeros((3, 3))
    rotation_matrix[:, 2] = -eye_direction / np.linalg.norm(eye_direction)  # View direction towards origin
    rotation_matrix[:, 0] = np.cross(rotation_matrix[:, 2], up)  # Camera-Right
    rotation_matrix[:, 0] /= np.linalg.norm(rotation_matrix[:, 0])
    rotation_matrix[:, 1] = np.cross(rotation_matrix[:, 2], rotation_matrix[:, 0])  # Camera-Down
    rotation_matrix = rotation_matrix.T

    # Apply roll rotation using Rodrigues' formula + set position accordingly:
    rodriguez = np.asarray([0, 0, 1]) * (roll_angle * np.pi / 180.0)
    angle_axis = scipy.linalg.expm(np.cross(np.eye(3), rodriguez))
    rotation_matrix = np.dot(angle_axis, rotation_matrix)

    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation_matrix
    transform_matrix[0:3, 3] = [0, 0, scipy.linalg.norm(camera_position)]
    return transform_matrix

from OpenGL.GL import *
from OpenGL.GLU import *
def read_texture(filename):
    import numpy 
    from PIL import Image
    img = Image.open(filename)
    img_data = numpy.array(list(img.getdata()), numpy.int8)
    textID = glGenTextures(1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return textID
