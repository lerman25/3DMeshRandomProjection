class Mesh:
    def __init__(self, vertices, faces, vertex_colors):
        self.vertices = vertices
        self.faces = faces
        self.vertex_colors = vertex_colors
    def print_mesh_meta(self):
        print(f"Vertices: {self.vertices.shape}")
        print(f"Faces: {self.faces.shape}")
        print(f"Vertex colors: {self.vertex_colors.shape}") 