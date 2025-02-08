import numpy as np
import copy
import scipy


class VirtualCamera:
    def __init__(self,K,R = np.zeros((3,3)),t = np.ones(3),D = 0,image_size = (640,480),clip_near = 0.001,clip_far = 10000.0) -> None:
        self._K = K
        self._R = R
        self._t = t
        self._D = D
        self._image_size = image_size
        self._clip_near = clip_near
        self._clip_far = clip_far
        self._projection_matrix = _convert_hz_intrinsic_to_opengl_projection(self)
    def get_K(self):
        return self._K
    def copy(self):
        return copy.deepcopy(self)
    def get_projection_matrix(self):
        return self._projection_matrix
    
    @staticmethod
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
    
    @staticmethod
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
    
def _convert_hz_intrinsic_to_opengl_projection(vc : VirtualCamera,x0=0,y0=0, flipy=False):
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
