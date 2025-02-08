class OpenGL_code:
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
        for key, value in OpenGL_code.__dict__.items():
            if not key.startswith("__"):
                self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")