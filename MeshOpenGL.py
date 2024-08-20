import numpy as np
from VirtualCamera import VirtualCamera
from vispy import app, gloo
import OpenGL.GL as gl
import OpenGL.GLU as glu
from FootballPitchMeshLoader import PitchMeshLoader
from OpenGLUtils import prepare_opengl,pass_mesh_to_opengl,look_at,fibonacci_sphere,random_point_on_upper_hemisphere
from OpenGLUtils import OpenGL_constant, read_texture
from PlotUtils import plot_image_grid
import matplotlib.pyplot as plt
import cv2
class MeshOpenGL:
    def __init__(self,mesh : PitchMeshLoader,vc : VirtualCamera) -> None:
        self._pitch = mesh
        self._vc = vc
        self.prepare()
        
        # Rendering with OpenGL
    def prepare(self):
        self._canvas = app.Canvas(show=False, size=self._vc._image_size)
        tex_shape = (self._vc._image_size[1], self._vc._image_size[0]) # coordinates are inverted in OpenGL
        # Texture to render the image into:
        self._image_texture = gloo.Texture2D(shape=tex_shape + (3,))
        

        # Corresponding FBO (frame buffer):
        self._fbo = gloo.FrameBuffer(self._image_texture, gloo.RenderBuffer(tex_shape))
        self._fbo.activate()
        gloo.set_state(depth_test=True, blend=False, cull_face=False)
        gloo.set_clear_color(self._vc._background_color)
        gloo.set_viewport(0, 0, *self._canvas.size)
        self._vertex_buffer,self._index_buffer = pass_mesh_to_opengl(self._pitch)
        self._gl_program = gloo.Program(OpenGL_constant.VERTEX_SHADER_BASIC, OpenGL_constant.FRAGMENT_SHADER_COLOR)
        self._model_matrix = np.eye(4, dtype=np.float32) # set at the origin
    def calculateMVP(self,camera_translation_vector,camera_rotation_matrix):
        self._yz_flip = np.eye(4, dtype=np.float32)
        self._yz_flip[1, 1], self._yz_flip[2, 2] = -1, -1

        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[:3, 3] = np.squeeze(camera_translation_vector)
        view_matrix[:3, :3] = camera_rotation_matrix

        # Converting it to OpenGL coordinate system:
        view_matrix = self._yz_flip.dot(view_matrix).T

        # Model-view matrix (projecting from object space to camera space):
        self._mv_matrix = np.dot(self._model_matrix, view_matrix)

        # Model-view-projection matrix (projecting from object space to image space):
        self._mvp_matrix = np.dot(self._mv_matrix, self._vc._projection_matrix)
    
    def draw(self,camera_translation_vector, camera_rotation_matrix,directional_light_vector,ambient_light):
        """
        Render and return color and optionally depth images of the mesh, from the chosen viewpoint.
        :param camera_translation_vector:   Camera position
        :param camera_rotation_matrix:      Camera rotation
        :param ambient_light:               Ambient light factor
        :param directional_light_vector:    Vector of directional light
        :return:                            RGB image
        """

    # MVP matrices:
        self.calculateMVP(camera_translation_vector, camera_rotation_matrix)

        # Clear previous content:
        gloo.clear(color=True, depth=True)

        # Bind mesh buffer to shader program:
        self._gl_program.bind(self._vertex_buffer)
        # Pass parameters to shader program:
        self._gl_program['u_mv'] = self._mv_matrix
        self._gl_program['u_mvp'] = self._mvp_matrix
        self._gl_program['u_light_position'] = directional_light_vector
        self._gl_program['u_light_ambient'] = ambient_light
        gl.glActiveTexture(gl.GL_TEXTURE0)  # Activate texture unit 0
        if self._pitch._texture:
            
            texture_0 = read_texture(self._pitch._texture)
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D,texture_0)
            img = cv2.imread(self._pitch._texture)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self._image_texture = gloo.Texture2D(img)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_0)
            self._gl_program['textureSampler'] = 0  # Pass the texture unit to the shader

        # Render:
        self._gl_program.draw('triangles',self._index_buffer)

        # Fetch rendered content from FBO:
        bgr = np.copy(gloo.read_pixels((0, 0, *self._vc._image_size))[..., :3])
        rgb = bgr[..., ::-1]
        rgb = cv2.flip(rgb, 0) 

        return rgb
    def generate_random_images(self,num_images=10,radius = 0.2):
        camera_positions = random_point_on_upper_hemisphere(radius=radius, samples=num_images)
        camera_transform_matrices = [look_at(camera_position, self._model_matrix[0:3, 3])
                           for camera_position in camera_positions]
        camera_positions = [camera_transform_matrix[0:3, 3] 
                            for camera_transform_matrix in camera_transform_matrices]
        camera_rotation_matrices = [camera_transform_matrix[0:3, 0:3]
                                    for camera_transform_matrix in camera_transform_matrices]

        self._rgb_images = []
        ret_val = []
        for camera_position, camera_rotation_matrix in zip(camera_positions, camera_rotation_matrices):

            # Randomize lighting confitions:
            ambient_light = np.random.uniform(0.9, 1)
            directional_light_vector = np.random.uniform(-1, 1, size=3)

            rgb_image = self.draw(camera_position, camera_rotation_matrix,directional_light_vector,ambient_light)
            self._rgb_images.append(rgb_image)
            ret_val.append((rgb_image,camera_position,camera_rotation_matrix))
        self._rgb_images = np.asarray(self._rgb_images)
        return ret_val
    def plot_generated_images(self):
        figure = plot_image_grid([self._rgb_images], transpose=True)
        plt.show()
        