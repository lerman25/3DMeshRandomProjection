from vispy import app, gloo
import OpenGL.GL as gl
from VirtualCamera import VirtualCamera as VC
from Mesh import Mesh
from OpenGLutils import pass_mesh_to_opengl, compute_mv_and_mvp
from OpenGLcode import OpenGL_code
app.use_app('PyGlet')  # Set backend (try e.g. "PyQt5" otherwise)
import numpy as np
from Scene import Scene

class Canvas:
    def __init__(self, image_size, background_color = (0,0,1)) -> None:
        self.image_size = image_size
        canvas = app.Canvas(show=False, size=image_size)
        tex_shape = (image_size[1], image_size[0]) # coordinates are inverted in OpenGL
        image_texture = gloo.Texture2D(shape=tex_shape + (3,))
        fbo = gloo.FrameBuffer(image_texture, gloo.RenderBuffer(tex_shape))
        fbo.activate()
        gloo.set_state(depth_test=True, blend=False, cull_face=False)
        gloo.set_clear_color(background_color)
        gloo.set_viewport(0, 0, *canvas.size)
    
    def load_mesh_on_canvas(self,mesh : Mesh):
        vertex_buffer, index_buffer = pass_mesh_to_opengl(mesh, 'a_position', 'a_color')
        self.vertex_buffer = vertex_buffer
        self.index_buffer = index_buffer
        self.mesh = mesh
        self.gl_program = gloo.Program(OpenGL_code.VERTEX_SHADER_BASIC, OpenGL_code.FRAGMENT_SHADER_COLOR)
        self.model_matrix = np.eye(4, dtype=np.float32) # set at the origin

    def draw(self,scene: Scene, camera_translation_vector, camera_rotation_matrix):
        """
        Render and return color and optionally depth images of the mesh, from the chosen viewpoint.
        :param camera_translation_vector:   Camera position
        :param camera_rotation_matrix:      Camera rotation
        :param ambient_light:               Ambient light factor
        :param directional_light_vector:    Vector of directional light
        :return:                            RGB image
        """

        # MVP matrices:
        mv_matrix, mvp_matrix = compute_mv_and_mvp(
            self.model_matrix, scene.get_vc().get_projection_matrix(), camera_translation_vector, camera_rotation_matrix)

        # Clear previous content:
        gloo.clear(color=True, depth=True)

        # Bind mesh buffer to shader program:
        self.gl_program.bind(self.vertex_buffer)

        # Pass parameters to shader program:
        self.gl_program['u_mv'] = mv_matrix
        self.gl_program['u_mvp'] = mvp_matrix
        self.gl_program['u_light_position'] = scene.get_light_direction_vector()
        self.gl_program['u_light_ambient'] = scene.get_ambient_light()

        # Render:
        self.gl_program.draw('triangles',self.index_buffer)

        # Fetch rendered content from FBO:
        bgr = np.copy(gloo.read_pixels((0, 0, *self.image_size))[..., :3])
        rgb = bgr[..., ::-1]
        return rgb
