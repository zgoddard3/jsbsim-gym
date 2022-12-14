import pygame as pg
import numpy as np
import moderngl as mgl

from quaternion import Quaternion

def load_shader(ctx : mgl.Context, vertex_filename, frag_filename):
    with open(vertex_filename) as f:
        vertex_src = f.read()
    with open(frag_filename) as f:
        frag_src = f.read()
    
    return ctx.program(vertex_shader=vertex_src, fragment_shader=frag_src)

def load_mesh(ctx : mgl.Context, program, filename):
    v = []
    vn = []
    vertices = []
    indices = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertex = [float(val) for val in values[1:4]]
                v.append(vertex)
            elif values[0] == 'vn':
                norm = [float(val) for val in values[1:4]]
                vn.append(norm)
            elif values[0] == 'vt':
                continue
            elif values[0] in ('usemtl', 'usemat'):
                continue
            elif values[0] == 'mtllib':
                continue
            elif values[0] == 'f':
                for val in values[1:]:
                    w = val.split('/')
                    vertex = np.hstack((v[int(w[0])-1], vn[int(w[2])-1]))
                    vertices.append(vertex)
                start = len(vertices) - len(values) + 1
                for i in range(start, len(vertices)-2):
                    indices.append([start, i+1, i+2])

    vbo = ctx.buffer(np.hstack(vertices).astype(np.float32).tobytes())
    ebo = ctx.buffer(np.hstack(indices).flatten().astype(np.uint32).tobytes())
    return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', index_buffer=ebo)

def perspective(fov, aspect, near, far):
    fov *= np.pi/180
    right = -np.tan(fov/2) * near
    top = -right / aspect
    return np.array([[near/right,0,0,0],
                     [0,near/top,0,0],
                     [0,0,(far+near)/(far-near),-2*far*near/(far-near)],
                     [0,0,1,0]], dtype=np.float32)

class Transform:
    def __init__(self):
        self._position = np.zeros(3)
        self._rotation = Quaternion()
        self.scale = 1
    
    @property
    def position(self):
        return self._position.copy()
    
    @position.setter
    def position(self, position):
        self._position[:] = position
    
    @property
    def x(self):
        return self._position[0]
    
    @x.setter
    def x(self, x):
        self._position[0] = x
    
    @property
    def y(self):
        return self._position[1]
    
    @y.setter
    def y(self, y):
        self._position[1] = y
    
    @property
    def z(self):
        return self._position[2]
    
    @z.setter
    def z(self, z):
        self._position[2] = z
    
    @property
    def rotation(self):
        return self._rotation.copy()
    
    @rotation.setter
    def rotation(self, rotation):
        self._rotation._arr[:] = rotation._arr
    
    @property
    def matrix(self):
        matrix = np.eye(4)
        matrix[:3,:3] = self._rotation.mat().dot(np.eye(3)*self.scale)
        matrix[:3,3] = self._position
        return matrix
    
    @property
    def inv_matrix(self):
        matrix = np.eye(4)
        matrix[:3,3] = -self._position
        scale = np.eye(4)
        scale[:3,:3] /= self.scale
        matrix = scale.dot(matrix)
        rot = np.eye(4)
        rot[:3,:3] = self.rotation.inv().mat()
        matrix = rot.dot(matrix)
        return matrix

    
class RenderObject:
    def __init__(self, vao):
        self.vao = vao

        self.color = 1.0, 1.0, 1.0

        self.transform = Transform()

        self.draw_mode = mgl.TRIANGLES
    
    def render(self):
        self.vao.program['model'] = tuple(np.hstack(self.transform.matrix.T))
        self.vao.program['color'] = self.color
        self.vao.render(self.draw_mode)

class Grid(RenderObject):
    def __init__(self, ctx : mgl.Context, program, n, spacing):
        super().__init__()
        low = -(n-1)*spacing/2
        high = -low
        vertices = []
        indices = []
        for i in range(n):
            vertices.append([low + spacing*i, 0, low])
            vertices.append([low + spacing*i, 0,  high])
            indices.append([i*2, i*2+1])
        for i in range(n):
            vertices.append([low, 0, low + spacing*i])
            vertices.append([high, 0, low + spacing*i])
            indices.append([n*2+i*2, n*2+i*2+1])
        vertices = np.hstack(vertices)
        indices = np.hstack(indices)
        vbo = ctx.buffer(vertices.astype(np.float32).tobytes())
        ebo = ctx.buffer(indices.astype(np.uint32).tobytes())
        self.vao = ctx.simple_vertex_array(program, vbo, 'aPos', index_buffer=ebo)
        self.draw_mode = mgl.LINES

class Viewer:
    def __init__(self, width, height, fps=30, headless=False):
        self.transform = Transform()
        self.width = width
        self.height = height
        self.fps = fps

        if headless:
            self.ctx = mgl.create_standalone_context()
            self.display = None
            self.clock = None
        else:
            pg.init()
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 3)
            self.display = pg.display.set_mode((width, height), pg.DOUBLEBUF | pg.OPENGL)
            self.ctx = mgl.create_context()
            self.clock = pg.time.Clock()
        
        self.ctx.enable(mgl.DEPTH_TEST)
        
        self.projection = perspective(90, width/height, .1, 1000.)
        
        self.prog = load_shader(self.ctx, "simple.vert", "simple.frag")
        self.prog['projection'] = tuple(np.hstack(self.projection.T))
        self.prog['lightDir'] = .6, -.8, 1.0

        self.unlit = load_shader(self.ctx, "simple.vert", "unlit.frag")
        self.unlit['projection'] = tuple(np.hstack(self.projection.T))
        self.set_view()

        self.objects = []
    
    def run(self):
        running = True
        t = 0
        while running:
            pg.event.pump()
            camera_speed = .3
            camera_rot = np.pi/36
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            
            self.callback()

            self.render()
            self.clock.tick(self.fps)
            t += 1


            self.scene.root.children[0].transform.rotation = Quaternion.from_euler(0, t*np.pi/self.fps, 0)

        self.close()
    
    def callback(self):
        pass

    def set_view(self, x=None, y=None, z=None, rotation=None):
        if x is not None:
            self.transform.x = x
        if y is not None:
            self.transform.y = y
        if z is not None:
            self.transform.z = z
        if rotation is not None:
            self.transform.rotation = rotation
        self.prog['view'] = tuple(np.hstack(self.transform.inv_matrix.T))
        self.unlit['view'] = tuple(np.hstack(self.transform.inv_matrix.T))

    def _get_frame(self):
        data = self.ctx.fbo.read()
        return np.array(bytearray(data)).reshape(self.height, self.width,3)[-1::-1,:,:]
    
    def render(self):
        self.ctx.clear(0.1, 0.5, 0.3, 1.0)

        for obj in self.objects:
            obj.render()

        pg.display.flip()
    
    def close(self):
        pg.quit()

if __name__ == "__main__":
    from numpy.linalg import inv
    trans = Transform()
    trans.position = -2,2,3
    trans.rotation = Quaternion.from_euler(-.5,-.2,.3)
    trans.scale = 5.0

    print(np.sum(np.abs(trans.inv_matrix) - np.abs(inv(trans.matrix))))