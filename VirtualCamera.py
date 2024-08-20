import numpy as np
import copy
def create_camera_matrix(fx,fy,cx,cy,skew = 0):
    """
    Initialize a camera matrix according to intrinsic parameters.
    :param fx:          Horizontal focal length (in px)
    :param fy:          Vertical focal length (in px)
    :param cx:          Horizontal principal point offset (in px)
    :param cy:          Vertical principal point offset (in px)
    :param skew:        (opt.) Axis skew factor
    :return:            Camera matrix
    """
    # Camera matrix:
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[0, 1] = skew
    return K

class VirtualCamera:
    def __init__(self,K,R = np.zeros((3,3)),t = np.ones(3),D = 0,image_size = (640,480),clip_near = 0.001,clip_far = 10000.0,background_color = (0,0,1)) -> None:
        self._K = K
        self._R = R
        self._t = t
        self._D = D
        self._image_size = image_size
        self._clip_near = clip_near
        self._clip_far = clip_far
        self._background_color = background_color
        self._projection_matrix = convert_hz_intrinsic_to_opengl_projection(self)
    def get_K(self):
        return self._K
    def copy(self):
        return copy.deepcopy(self)
    
def convert_hz_intrinsic_to_opengl_projection(vc : VirtualCamera,x0=0,y0=0, flipy=False):
    """
    Convert camera parameter (Hartley-Zisserman intrinsic matrix) into a projection matrix for OpenGL.
    Snippet by Andrew Straw
    (https://gist.github.com/astraw/1341472/c5f8aba7f81431967d1fc9d954ae20822c616c17#file-calib_test_utils-py-L67)
    :param K:       Camera matrix
    :param x0:      Camera horizontal image origin (typically 0)
    :param y0:      Camera vertical image origin (typically 0)
    :param width:   Canvas width
    :param height:  Canvas height
    :param znear:   Clip-near value
    :param zfar:    Clip-far value
    :param flipy:   Flag to True if images should be rendered upside-down (to match other pixel coordinate systems)
    :return:        Camera projection matrix
    """
    znear = float(vc._clip_near)
    zfar = float(vc._clip_far)
    depth = zfar - znear
    q = -(zfar + znear) / depth
    qn = -2 * (zfar * znear) / depth
    K = vc._K
    width = vc._image_size[0]
    height = vc._image_size[1]

    if not flipy:
        proj = np.array([[2 * K[0, 0] / width, -2 * K[0, 1] / width, (-2 * K[0, 2] + width + 2 * x0) / width, 0],
                         [0, -2 * K[1, 1] / height, (-2 * K[1, 2] + height + 2 * y0) / height, 0],
                         [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes.
                         [0, 0, -1, 0]])  # This row is also standard glPerspective.
    else:
        proj = np.array([[2 * K[0, 0] / width, -2 * K[0, 1] / width, (-2 * K[0, 2] + width + 2 * x0) / width, 0],
                         [0, 2 * K[1, 1] / height, (2 * K[1, 2] - height + 2 * y0) / height, 0],
                         [0, 0, q, qn],  # This row is standard glPerspective and sets near and far planes.
                         [0, 0, -1, 0]])  # This row is also standard glPerspective.
    return proj.T