import sys
sys.path.append('../')


from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4
from shiboken2 import VoidPtr
from scripts.MeshAndShaders import Mesh
from scripts.BranchGeometry import BranchGeometry
# from scripts.InterfaceLayout import MainWindow, TestWindow
# from PySide2.shiboken2 import VoidPtr


import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
# from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo
import pywavefront
import numpy as np
import ctypes                 # to communicate with c code under the hood




##############################################################################################################
#
#  Code inspiration taken from:
#  https://github.com/D-K-E/pyside-opengl-tutorials/blob/master/tutorials/01-triangle/TriangleTutorial.ipynb
#  https://github.com/PyQt5/Examples/blob/master/PySide2/opengl/hellogl2.py#L167 
#
#
#   DESCRIPTION: Loads the interface window to display the tree with all the buttons, sliders, and windows
##############################################################################################################

class GLWidget(QOpenGLWidget): 

    xRotationChanged = Signal(int)
    yRotationChanged = Signal(int)
    zRotationChanged = Signal(int)

    def __init__(self, tree_fname=None):
        super().__init__()
        # self.parent = parent

        # if the 
        if tree_fname is not None:
            self.mesh = Mesh(tree_fname, None)
        else:
            self.mesh = Mesh(None, BranchGeometry("data"))
        # self.mesh = Mesh('tree_files/side2_branch_1.obj', None)
        # self.mesh = Mesh(None, BranchGeometry("data"))
        # self.mesh = Mesh(None, BranchGeometry("data"))

        # self.mesh = Mesh('tree_files/testSplineBranch.obj')
        # self.mesh = Mesh('tree_files/exemplarTree.obj')
        self.background_color = QtGui.QColor(0, 59, 111) 


        # for the mouse key events
        self.lastPos = 0

        # opengl data
        self.context = QOpenGLContext()
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.program = QOpenGLShaderProgram()

        # self.tree_color = QVector4D(0.5, 0.25, 0.0, 0.0) # brown color
        self.tree_color = QtGui.QColor(0, 139, 69, 19)
        
        self.world = QMatrix4x4() # the world view
        self.proj = QMatrix4x4() # where to look
        self.camera = QMatrix4x4() # the camera position

        self.xRot = 0
        self.yRot = 0
        self.zRot = 0

        # for binding the color
       
        

    def xRotation(self):
        return self.mesh.xRot
    
    def yRotation(self):
        return self.mesh.yRot
    
    def zRotation(self):
        return self.mesh.zRot
            
    def xScale(self):
        return self.mesh.sx
    
    def yScale(self):
        return self.mesh.sy

    def zScale(self):
        return self.mesh.sz

    
    def normalizeScale(self, scale):
        while scale < 0:
            scale += 100 * 10
        while scale > 100 * 10:
            scale -= 100 * 10
        return scale
    

    def setDiameterScale(self, scale):
        scale = self.normalizeScale(scale)
        if scale != self.mesh.diameter:
            self.mesh.diameter = scale
            self.mesh.diameterChanged.emit(scale)
            self.update()

    
    def setXScale(self, scale):
        scale = self.normalizeScale(scale)
        if scale != self.mesh.sx:
            self.mesh.sx = scale
            self.mesh.xScaleChanged.emit(scale)
            self.update()
    

    # comparing for the length of the branch
    def setLength(self, length):
        length = self.normalizeScale(length)
        # print("Length:", length)
        if length != self.mesh.branchLength:
            self.mesh.branchLength = length
            self.mesh.lengthChanged.emit(length)
            self.update()

    
    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle


    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.mesh.xRot:
            self.mesh.xRot = angle
            self.mesh.xRotationChanged.emit(angle)
            # self.emit(SIGNAL("xRotationChanged(int)"), angle)
            self.update()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.mesh.yRot:
            self.mesh.yRot = angle
            self.mesh.yRotationChanged.emit(angle)
            # self.emit(SIGNAL("yRotationChanged(int)"), angle) # telling the system what exactly has changed
            self.update()
    
    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.mesh.zRot:
            self.mesh.zRot = angle
            self.mesh.zRotationChanged.emit(angle)
            # self.emit(SIGNAL("zRotationChanged(int)"), angle)
            self.update()
    
    
    def mousePressEvent(self, event) -> None:
        self.startPos = QPoint(event.pos())      # returns the last position of the mouse when clicked
        print("Start mouse position", self.startPos)
        # need to look for when the mouse is pressed then released

        # return super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event) -> None:
        self.lastPos = QPoint(event.pos())
        print("Last mouse position", self.lastPos)    
    
    
    def getGlInfo(self):
        "Get opengl info"
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
            """.format(
                gl.glGetString(gl.GL_VENDOR),
                gl.glGetString(gl.GL_RENDERER),
                gl.glGetString(gl.GL_VERSION),
                gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
            )
        return info
    
    # Basic Function needed by OpenGL
    def initializeGL(self):
        print('gl initial')
        print(self.getGlInfo())

        # create context 
        self.context.create()
        # if the close signal is given we clean up the ressources as per defined above
        self.context.aboutToBeDestroyed.connect(self.cleanUpGl)  


       # initialize functions
        funcs = self.context.functions()  # we obtain functions for the current context
        funcs.initializeOpenGLFunctions() # we initialize functions
        funcs.glClearColor(self.background_color.redF(), self.background_color.greenF(), self.background_color.blueF(), self.background_color.alphaF()) # the color that will fill the frame when we call the function
        # for cleaning the frame in paintGL

        funcs.glEnable(gl.GL_DEPTH_TEST) # enables depth testing so things are rendered correctly

        # INITIALZE OUR SHADERS
        vshader, fshader = self.mesh.initializeShaders()

        # creating shader program
        self.program = QOpenGLShaderProgram(self.context)
        self.program.addShader(vshader)  # adding vertex shader
        self.program.addShader(fshader)  # adding fragment shader

        # bind attribute to a location in the shader
        self.program.bindAttributeLocation("vertexPos", 0) # getting the vertex position for shader.vert

        # link the shader program
        isLinked = self.program.link()
        print("Shader program is linked: ", isLinked)

        # bind the program --> activates it!
        self.program.bind()

        # bind the location of the color for the shader.frag to use
        colorLoc = self.program.uniformLocation("color")
        print(f"Color Location {colorLoc}")
        self.program.setUniformValue(colorLoc, self.tree_color)
        


    
        # self.mvMatrixLoc = self.mesh.program.uniformLocation("mvMatrix")

        # self.mesh.initializeMeshArrays()


        # INITIALIZING VALUES FOR ARRAYS 
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

        # self.setupMeshAttribArrays()

        # SETUP MESH ATTRIBUTE ARRAYS
        float_size = ctypes.sizeof(ctypes.c_float)
        
        # allocate buffer space
        self.vbo.allocate(self.mesh.vertices.tobytes(), 
                          float_size * self.mesh.vertices.size) # TO CHANGE WITH BRANCHES ADDED AND ABILITY TO DRAW

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

        self.camera.setToIdentity() # sets the camera matrix to the identity
        self.camera.translate(0, 0, -2)

        self.program.release()
        vaoBinder = None


        # self.camera.setToIdentity()
        # self.camera.translate(0, 0, -4)


    
    def setupVertexAttribs(self):
        self.vbo.bind()
        funcs = self.context.functions()  # self.context.currentContext().functions()  
        funcs.glEnableVertexAttribArray(0)

        float_size = ctypes.sizeof(ctypes.c_float)
        null = VoidPtr(0)
        funcs.glVertexAttribPointer(0,                 # where the array starts
                                3,                     # how long the vertex point is i.e., (x, y, z)
                                gl.GL_FLOAT,           # is a float
                                gl.GL_FALSE,
                                3 * float_size,        # size in bytes to the next vertex
                                null)                  # where the data is stored (starting position) in memory
        
        self.vbo.release()

    

    # Basic function required by OpenGL
    def resizeGL(self, width: int, height: int):
        funcs = self.context.functions()
        funcs.glViewport(0, 0, width, height)

        # # funcs.glMatrixMode(gl.GL_PROJECTION) # used to define the vieweing volume contianing the projection transformation
        # gl.glMatrixMode(gl.GL_PROJECTION) 
        self.mesh.proj.setToIdentity()
        
        # # funcs.glLoadIdentity()
        
        aspect = width / float(height)        
        # GLU.gluPerspective(45.0, aspect, 1.0, 100.0) # defines the viewing frustrum
        self.mesh.proj.perspective(45.0, aspect, 0.01, 100)
        # # funcs.glMatrixMode(gl.GL_MODELVIEW)
        # gl.glMatrixMode(gl.GL_MODELVIEW)

    # Cleans up the buffers upon exiting the app
    def cleanUpGl(self):
        self.context.makeCurrent(self)      # make context to release the current one
        self.vbo.destroy()                  # destroy the buffer
        del self.program                    # delete the shader program
        self.program = None                 # change pointed reference
        self.doneCurrent()                  # no current context for current thread

    # Basic function needed for OpenGL
    def paintGL(self):
        funcs = self.context.functions()
        # clean up what was drawn in the previous frame
        funcs.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) 

        # self.program.setUniformValue()

        # scaling goes here as well
        
        # self.mesh.drawMesh()

        # funcs = self.context.functions()
        # DRAW MESH
        # load the identity to do the scaling of the function
        # self.world.setToIdentity()
        # self.world.rotate(self.xRot/16, 1, 0, 0)
        # self.world.rotate(self.yRot/16, 0, 1, 0)
        # self.world.rotate(self.zRot/16, 0, 0, 1)
        
        # # self.world.scale(self.sx/10, self.diameter/10, self.diameter/10)  # self.sy, self.sz
        # self.world.scale(self.sx, self.sy, self.sz)

        # need to update the branch to draw
        # self.update_vertices()
        # self.setupMeshAttribArrays()

        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.bind()

        # setting the values of the projecton matrix and the movement matrix
        # self.program.setUniformValue(self.projMatrixLoc, self.proj)
        # self.program.setUniformValue(self.mvMatrixLoc, self.camera * self.world)

        # INTRODUCES CAMERA MOVEMENTS
        # self.program.setUniformValue("projMatrix", self.proj)
        # self.program.setUniformValue("mvMatrix", self.camera * self.world)


        # normalMatrix = self.world.normalMatrix()
        # self.program.setUniformValue(self.normalMatrixLoc, normalMatrix)

        funcs.glDrawArrays(gl.GL_TRIANGLES,                     # telling it to draw triangles between vertices
                           0,                                   # where the vertices starts
                           self.mesh.vertex_count)              # how many vertices to draw from the mesh

        self.program.release()
        self.vao.release()
        vaoBinder = None
        
   



# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     # window = MainWindow()
#     window.showMaximized()
#     sys.exit(app.exec_())