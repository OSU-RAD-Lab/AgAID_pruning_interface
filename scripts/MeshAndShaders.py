import sys
sys.path.append('../')


from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4
from shiboken2 import VoidPtr
# from PySide2.shiboken2 import VoidPtr

# from PyQt6 import QtCore      # core Qt functionality
# from PyQt6 import QtGui       # extends QtCore with GUI functionality
# from PyQt6 import QtOpenGL    # provides QGLWidget, a special OpenGL QWidget

from scripts.BranchGeometry import BranchGeometry     # personal class used to help participants manipulate branches

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
# from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo
import pywavefront
import numpy as np
import ctypes                 # to communicate with c code under the hood


class Shader:
    def __init__(self, shaderType: str, shaderName: str, shaderPath: str):
        self.shader = None
        self.shaderType = shaderType
        self.shaderName = shaderName
        self.shaderPath = shaderPath

        # shaderSource = self.shaders[shaderType]  # gets the source code from above
        if shaderType == "vertex":
            self.shader = QOpenGLShader(QOpenGLShader.Vertex)
        else:
            self.shader = QOpenGLShader(QOpenGLShader.Fragment)

        isCompiled = self.shader.compileSourceFile(shaderPath)

        if isCompiled is False:
            print(self.shader.log())
            raise ValueError(
                "{0} shader {2} known as {1} is not compiled".format(shaderType, shaderName, shaderPath)
            )
        
    def getShader(self):
        return self.shader



class Mesh(QWidget):
    # for the rotation of the widget window
    xRotationChanged = Signal(int)
    yRotationChanged = Signal(int)
    zRotationChanged = Signal(int)

    # might remove these depending on the BranchGeometry.py
    xScaleChanged = Signal(int)
    yScaleChanged = Signal(int)
    zScaleChanged = Signal(int)

    #
    diameterChanged = Signal(int)
    lengthChanged = Signal(int)
    branchCurveChanged = Signal(int)

    def __init__(self, fname=None, branch=None):
        super().__init__()

        # Rotation values
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0

        self.sx = 0.1       # gets the length of the branch
        self.diameter = 0.1

        self.sy = 0.1       # gets the diameter
        self.sz = 0.1       # gets the diameter

        # basic values for the branch
        self.branchCurve = 0.1              # how much curve the branch will have 
        self.branchLength = 0.3             # the length of the bud
        self.startR = 0.03                  # the starting radius of the branch
        self.endR = self.startR/2           # the end radius of the branch

        # getting pointers to the locations of the variables
        self.projMatrixLoc = 0
        self.mvMatrixLoc = 0
        self.normalMatrixLoc = 0

        self.proj = QMatrix4x4() # where to look
        self.camera = QMatrix4x4() # the camera position
        self.world = QMatrix4x4() # the world view
        self.model = QMatrix4x4()


        self.fname = fname
        self.branch = branch
        self.mesh = None
        self.vertices = [] 
        self.vertex_count = 0 

        # loading the shaders
        self.vshader = None
        self.fshader = None
        
        # loading and getting all the vertices information
        if self.fname is not None:
            self.mesh = self.load_mesh(fname)
            self.meshFaces = self.mesh.mesh_list[0].faces # Gives us the triangles that are drawn by the system\

            # get the vertices in the correct order based on faces
            self.meshVertices = self.get_mesh_vertex()
            self.mVertex_count = len(self.vertices) / 3 

            self.vertices.extend(self.meshVertices)
            self.vertex_count += self.mVertex_count
        

        if self.branch is not None:
            self.branchFaces = self.branch.branch_faces()
            self.branchVertices = self.branch.branch_vertices() 
            self.bVertex_count = len(self.branchVertices) / 3  

            self.vertices.extend(self.branchVertices)
            self.vertex_count += self.bVertex_count

            self.branchCurve = self.branch.branchCurve              # how much curve the branch will have 
            self.branchLength = self.branch.branchLength             # the length of the bud
            self.startR = self.branch.startR                # the starting radius of the branch
            self.endR = self.startR/2           # the end radius of the branch

            # extend the vertex list to include these vertices and the count
            self.vertices.extend(self.branchVertices)
            self.vertex_count += self.bVertex_count

        self.vertices = np.array(self.vertices)

        self.tree_color = QVector4D(0.5, 0.25, 0.0, 0.0) # brown color
       
        # opengl data
        self.context = QOpenGLContext()
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.program = QOpenGLShaderProgram()
        # self.vao.create()
        # vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)         
        
    
    # recapturing the vertices for the branch assuming anything has shifted since creating the branch

    def update_vertices(self):
        # append all vertices to the list and increment the counter accordingly
        self.vertices = []
        self.vertex_count = 0

        if self.mesh is not None:
            self.vertices.extend(self.meshVertices)
            self.vertex_count += self.mVertex_count

        if self.branch is not None:
            self.branch.branchLength = self.branchLength
            self.branch.branchCurve = self.branchCurve
            self.branch.startR = self.startR
            self.branch.endR = self.endR

            # recreate the branch with the new dimensions
            self.branch.create_branch()
            self.branchFaces = self.branch.branch_faces()
            self.branchVertices = self.branch.branch_vertices() 
            self.bVertex_count = len(self.branchVertices) / 3 

            self.vertices.extend(self.branchVertices)
            self.vertex_count += self.bVertex_count
        
        self.vertices = np.array(self.vertices)
        # print("updated vertices", self.vertices[:20])

    
    def load_mesh(self, fname):
        mesh = pywavefront.Wavefront(fname, collect_faces = True)
        # print(mesh)
        return mesh

    # Uses the faces information to extract the correct ordering for drawing triangles based on the vertex location
    def get_mesh_vertex(self):
        vertices = []
        for face in self.meshFaces:
            for vertex in face:
                vertices.extend(self.mesh.vertices[vertex]) # gets the exact vertex point and adds it to the list of values
                # print("Vertex {0} gets point {1}".format(v2, self.mesh.vertices[v2]))
        # print("Vertices\n", vertices[:10])
        return np.array(vertices, dtype=ctypes.c_float)
    
    
    def initializeShaders(self):
        shaderName = "tree"
        self.vshader = Shader(shaderType="vertex", shaderName=shaderName, shaderPath='../shaders/shader.vert').getShader()
        self.fshader = Shader(shaderType="fragment", shaderName=shaderName, shaderPath='../shaders/shader.frag').getShader()

        # creating shader program
        self.program = QOpenGLShaderProgram(self.context)
        self.program.addShader(self.vshader)  # adding vertex shader
        self.program.addShader(self.fshader)  # adding fragment shader

        # bind attribute to a location in the shader
        self.program.bindAttributeLocation("vertexPos", 0) # getting the vertex position

        # link the shader program
        isLinked = self.program.link()
        print("Shader program is linked: ", isLinked)

        # bind the program --> activates it!
        self.program.bind()

        # specify uniform value
        # self.colorLoc = self.program.uniformLocation("color") 
        # self.projMatrixLoc = self.program.uniformLocation("projMatrix")
        # self.mvMatrixLoc = self.program.uniformLocation("mvMatrix")


        # Connects the attributes to their coresponding values in the shaders files
        colorLoc = self.program.uniformLocation("color")     # shader.frag
        self.projMatrixLoc = self.program.uniformLocation("projMatrix") # shader.vert
        self.mvMatrixLoc = self.program.uniformLocation("mvMatrix") # shader.vert

        print(f"Color Location: {colorLoc}\nProject Matrix Location: {self.projMatrixLoc}\nMv Matrix Location: {self.mvMatrixLoc}")
        # self.normalMatrixLoc = self.program.uniformLocation("normalMatrix")
        
        # notice the correspondance of the
        # name color in fragment shader
        # we also obtain the uniform location in order to 
        # set value to it
        self.program.setUniformValue(colorLoc, self.tree_color) # for shader.frag
        # notice the correspondance of the color type vec4 
        # and the type of triangleColor
        
        


    def initializeMeshArrays(self):
        # create vao and vbo
        # vao
        isVao = self.vao.create()
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        # vbo
        isVbo = self.vbo.create()
        isBound = self.vbo.bind()

        # float_size = ctypes.sizeof(ctypes.c_float)

        # # allocate buffer space
        # self.vbo.allocate(self.vertices.tobytes(), 
        #                   float_size * self.vertices.size) # TO CHANGE WITH BRANCHES ADDED

        # check if vao and vbo are created
        print('vao created: ', isVao)
        print('vbo created: ', isVbo)
        print('vbo bound: ', isBound)

        self.setupMeshAttribArrays()

        self.camera.setToIdentity() # sets the camera matrix to the identity
        self.camera.translate(0, 0, -2)

        self.program.release()
        vaoBinder = None

    
    def setupMeshAttribArrays(self):

        float_size = ctypes.sizeof(ctypes.c_float)
        
        # allocate buffer space
        self.vbo.allocate(self.vertices.tobytes(), 
                          float_size * self.vertices.size) # TO CHANGE WITH BRANCHES ADDED AND ABILITY TO DRAW

        self.vbo.bind()
        funcs = self.context.functions()  # self.context.currentContext().functions()  
        funcs.glEnableVertexAttribArray(0)

        float_size = ctypes.sizeof(ctypes.c_float)
        null = VoidPtr(0)
        funcs.glVertexAttribPointer(0,                     # where the array starts
                                    3,                     # how long the vertex point is i.e., (x, y, z)
                                    gl.GL_FLOAT,           # is a float
                                    gl.GL_FALSE,
                                    3 * float_size,        # size in bytes to the next vertex (x, y, z)
                                    null)                  # where the data is stored (starting position) in memory
        
        self.vbo.release()
    
    
    def drawMesh(self):
        # drawing code
        funcs = self.context.functions()
        
        # load the identity to do the scaling of the function
        self.world.setToIdentity()
        self.world.rotate(self.xRot/16, 1, 0, 0)
        self.world.rotate(self.yRot/16, 0, 1, 0)
        self.world.rotate(self.zRot/16, 0, 0, 1)
        
        # self.world.scale(self.sx/10, self.diameter/10, self.diameter/10)  # self.sy, self.sz
        self.world.scale(self.sx, self.sy, self.sz)

        # need to update the branch to draw
        self.update_vertices()
        self.setupMeshAttribArrays()

        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.bind()

        # setting the values of the projecton matrix and the movement matrix
        self.program.setUniformValue(self.projMatrixLoc, self.proj)
        self.program.setUniformValue(self.mvMatrixLoc, self.camera * self.world)
        # normalMatrix = self.world.normalMatrix()
        # self.program.setUniformValue(self.normalMatrixLoc, normalMatrix)

        funcs.glDrawArrays(gl.GL_TRIANGLES,                     # telling it to draw triangles between vertices
                           0,                                   # where the vertices starts
                           self.vertex_count)                   # how many vertices to draw

        self.program.release()
        self.vao.release()
        vaoBinder = None
    
    
    def cleanUpMeshGL(self):
        self.context.makeCurrent(self)      # make context to release the current one
        self.vbo.destroy()                  # destroy the buffer
        del self.program                    # delete the shader program
        self.program = None                 # change pointed reference
        self.doneCurrent()                  # no current context for current thread

    def destroy(self):
        gl.glDeleteVertexArrays(1, (self.vao,))
        gl.glDeleteBuffers(1, (self.vbo,))