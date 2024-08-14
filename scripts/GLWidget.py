#!/usr/bin/env python3

import sys
sys.path.append('../')


from PySide2 import QtCore, QtGui, QtOpenGL
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4, QSurfaceFormat
from shiboken2 import VoidPtr
from scripts.MeshAndShaders import Mesh
from scripts.BranchGeometry import BranchGeometry
# from scripts.InterfaceLayout import MainWindow, TestWindow
# from PySide2.shiboken2 import VoidPtr


import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo

# 
# from PyQt5.QtGui import QSurfaceFormat, QOpenGLVersionProfile
# from PyQt5.QtOpenGL import QGLContext #QOpenGLVersionProfile

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
# QOpenGLWidget
# QtOpenGL.QGLWidget
class GLWidget(QOpenGLWidget): 

    xRotationChanged = Signal(int)
    yRotationChanged = Signal(int)
    zRotationChanged = Signal(int)

    def __init__(self, parent=None):
        # super().__init__()
        self.xsize = 512
        self.ysize = 512

        self.parent = parent
        # QtOpenGL.QGLWidget.__init__(self, parent)
        QOpenGLWidget.__init__(self, parent)
        
        self.background_color = QtGui.QColor(0, 59, 111) 
        # Create a context
        self.context = QOpenGLContext()

        # Set the surface format for the context

        # self.fmt = QSurfaceFormat()
        # self.fmt.setProfile(QSurfaceFormat.CoreProfile)
        # QSurfaceFormat.setDefaultFormat(self.fmt)


        # Initialize program for shaders, vao, and vbo for arrays
        self.program = QOpenGLShaderProgram()
        self.vao = QOpenGLVertexArrayObject()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        # for drawing the different lines  
        self.draw_program = QOpenGLShaderProgram()
        self.draw_vao = QOpenGLVertexArrayObject()
        self.draw_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)

        self.draw_lines = []

        # Initial testing vertices that draw a triangle
        # self.mesh = Mesh("../tree_files/exemplarTree.obj", None)
        # self.vertex = np.array(self.mesh.vertices, dtype = ctypes.c_float)

        self.vertex = np.array([-0.5, -0.5, 0.0,
                         0.5, -0.5, 0.0,
                         0.0, 0.5, 0.0], dtype=ctypes.c_float)
        
        self.FLOATSIZE = ctypes.sizeof(ctypes.c_float)

        # Shaders for the program to use
        self.VERTEX_SHADER = '''
            #version 140

            attribute highp vec3 vertexPos;

            void main()
            {
                gl_Position = vec4(vertexPos, 1.0);
            }    
        '''

        self.FRAGMENT_SHADER = '''
            # version 140

            uniform mediump vec4 color;
            out vec4 fragColor;

            void main()
            {
                fragColor = color;
                // fragColor = vec4(0.5, 0.25, 0.0, 0.0);
            }
        '''
        self.tree_color = [0.5, 0.25, 0.0, 0.0]
        self.triangle_color = [1.0, 0.0, 1.0, 0.0]
        
        self.proj = QMatrix4x4() # where to look
        self.camera = QMatrix4x4() # the camera position
        self.world = QMatrix4x4() # the world view
        # self.model = QMatrix4x4()

        self.xRot = 0
        self.yRot = 0
        self.zRot = 0

        self.width = 0
        self.height = 0
        self.startPose = QPoint()
        self.lastPose = QPoint()
        
        # for binding the color       

    # def xRotation(self):
    #     return self.mesh.xRot
    
    # def yRotation(self):
    #     return self.mesh.yRot
    
    # def zRotation(self):
    #     return self.mesh.zRot
            
    # def xScale(self):
    #     return self.mesh.sx
    
    # def yScale(self):
    #     return self.mesh.sy

    # def zScale(self):
    #     return self.mesh.sz

    
    def normalizeScale(self, scale):
        while scale < 0:
            scale += 100 * 10
        while scale > 100 * 10:
            scale -= 100 * 10
        return scale
    

    # def setDiameterScale(self, scale):
    #     scale = self.normalizeScale(scale)
    #     if scale != self.mesh.diameter:
    #         self.mesh.diameter = scale
    #         self.mesh.diameterChanged.emit(scale)
    #         self.update()

    
    # def setXScale(self, scale):
    #     scale = self.normalizeScale(scale)
    #     if scale != self.mesh.sx:
    #         self.mesh.sx = scale
    #         self.mesh.xScaleChanged.emit(scale)
    #         self.update()
    

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
        self.context = QOpenGLContext()
        if not self.context.create():
            raise Exception("Could not create a GL Context")
        self.context.aboutToBeDestroyed.connect(self.cleanUpGl)
        
        if not self.context.makeCurrent(self):
            raise Exception("makeCurrent() failed")
        
        
        # create the functions to call gl commands
        funcs = self.context.functions()
        funcs.initializeOpenGLFunctions()

        funcs.glClearColor(1, 1, 1, 0)  

        # create the program shader and bind vertex values to a location
        self.program = QOpenGLShaderProgram(self.context)
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.VERTEX_SHADER)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.FRAGMENT_SHADER)

        self.program.bindAttributeLocation("vertexPos", 0)

        if not self.program.link():
            print("Could not link program")

        # bind the program --> activates it
        self.program.bind()

        # set the uniform value of color to the variable we have defined
        colorLoc = self.program.uniformLocation("color")
        if colorLoc < 0:
            print("Failed to set a location for color")
        self.program.setUniformValue(colorLoc, self.tree_color[0], self.tree_color[1], self.tree_color[2], self.tree_color[3])

        # create the vao binder
        if not self.vao.create():
            print("Could not create a vao array")
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)
        
        # create and bind the vbo
        if not self.vbo.create():
            print("Could not create a vbo array")
        if not self.vbo.bind():
            print("Could not bind the vbo array")
        

        # Allocate the vertex buffer object with the size of our vertices (in bytes)
        self.vbo.allocate(self.vertex.tobytes(), self.FLOATSIZE * self.vertex.size)

        funcs.glEnableVertexAttribArray(0)
        nullptr = VoidPtr(0)
        funcs.glVertexAttribPointer(0, 
                                    3,
                                    int(gl.GL_FLOAT),
                                    int(gl.GL_FALSE),
                                    3 * self.FLOATSIZE,
                                    nullptr)
        
        # clean up the vbo and program
        self.vbo.release()
        self.program.release()
        vaoBinder = None
        self.vao.release()


        # # REPEAT FOR SMALLER TRIANGLE
        self.draw_program = QOpenGLShaderProgram(self.context)
        self.draw_program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.VERTEX_SHADER)
        self.draw_program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.FRAGMENT_SHADER)

        self.draw_program.bindAttributeLocation("vertexPos", 0)
        
        
        if not self.draw_program.link():
            print("Could not link program")

        # bind the program --> activates it
        self.draw_program.bind()

        triangleColorLoc = self.draw_program.uniformLocation("color")
        if triangleColorLoc < 0:
            print("Failed to set location for smaller triangle color")
        self.draw_program.setUniformValue(triangleColorLoc, self.triangle_color[0], self.triangle_color[1], self.triangle_color[2], self.triangle_color[3])

              
        # create the vao binder for drawing arrays
        if not self.draw_vao.create():
            print("Could not create a draw vao array")
        vaoDrawBinder = QOpenGLVertexArrayObject.Binder(self.draw_vao)
        vaoDrawBinder = None
        self.draw_vao.release()
        # # create and bind the vbo
        # if not self.draw_vbo.create():
        #     print("Could not create a draw vbo array")
        # if not self.draw_vbo.bind():
        #     print("Could not bind the draw vbo array")
        
        
        # self.draw_vertex = np.array(self.draw_vertex, dtype=ctypes.c_float)
        # Allocate the vertex buffer object with the size of our vertices (in bytes)

        # # Changed to draw_vertex
        # self.draw_vbo.allocate(self.small_vertex.tobytes(), 
        #                       self.FLOATSIZE * self.small_vertex.size)

        # funcs.glEnableVertexAttribArray(0)
        # nullptr = VoidPtr(0)
        # funcs.glVertexAttribPointer(0,                      # index
        #                             3,                      # size (i.e., x, y, z)
        #                             int(gl.GL_FLOAT),       # type
        #                             int(gl.GL_FALSE),       # normalized?
        #                             3 * self.FLOATSIZE,          # stride
        #                             nullptr)                # pointer to starting position
        

        
        # # clean up the vbo and program
        # self.draw_vbo.release()
        # self.draw_program.release()  
        # vaoDrawBinder = None
       
        # self.camera.setToIdentity() # sets the camera matrix to the identity
        # self.camera.translate(0, 0, -2)

        # gl.glDeleteProgram(self.program)
    
    

    # Basic function required by OpenGL
    def resizeGL(self, width: int, height: int):
        self.width = width
        self.height = height
        funcs = self.context.functions()
        funcs.glViewport(0, 0, width, height)

        # gl.glViewport(0, 0, width, height)
        # funcs.glMatrixMode(gl.GL_PROJECTION)
        # funcs.glLoadIdentity()
        gl.glLoadIdentity()
        aspect = width / float(height)      
        GLU.gluPerspective(45.0, aspect, 1.0, 100.0) # defines the viewing frustrum
        # self.mesh.proj.perspective(45.0, aspect, 0.01, 100)
        # # funcs.glMatrixMode(gl.GL_MODELVIEW)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    # Cleans up the buffers upon exiting the app
    def cleanUpGl(self):
        self.context.makeCurrent(self)
        funcs = self.context.functions()
        # self.context.makeCurrent(self)      # make context to release the current one
        self.vbo.destroy()                  # destroy the buffer
        del self.program                    # delete the shader program
        self.program = None                 # change pointed reference
        self.context.doneCurrent()                  # no current context for current thread

    # Basic function needed for OpenGL
    def paintGL(self):
        # Create the functions to use
        if not self.context.makeCurrent(self):
            raise Exception("makeCurrent failed (paintGL)")

        funcs = self.context.functions()
        funcs.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Bind the vertex attribute array to call for drawing
        vaoBinder = QOpenGLVertexArrayObject.Binder(self.vao)

        # bind the program to use the shaders defined earlier in initializeGL
        self.program.bind() # equivalent to gl.glUseProgram(self.program)

        # draw the tree
        funcs.glDrawArrays(gl.GL_TRIANGLES,     # how to draw
                           0,                   # where to start drawing/location of values
                           int(self.vertex.size/3))                   # size

        # release the program
        self.program.release()
        vaoBinder = None
        self.vao.release()

        self.context.swapBuffers(self)
        self.context.doneCurrent()

        # print(f"Draw lines in paintGL {self.draw_lines}")
        # self.mouseDraw()
        # self.drawLines()

        
    def mousePressEvent(self, event) -> None:
        self.startPose = QPoint(event.pos()) # returns the last position of the mouse when clicked
        # return super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event) -> None:
        self.lastPose = QPoint(event.pos())
        self.mouseDraw() 

    def mouseMoveEvent(self, event) -> None:
        pass

    def mouseDraw(self):
        diffX = np.abs(self.startPose.x() - self.lastPose.x())
        diffY = np.abs(self.startPose.y() - self.lastPose.y())
        if diffX < (self.width / 20) and diffY < (self.height / 20):
            print(f"Difference between drawing points is too small {diffX} and {diffY}")
        else:
            # Convert the coordinates between [-1, 1]
            startX = ((self.startPose.x() / self.width) - 0.5) * 2
            startY = ((self.startPose.y() / self.height) - 0.5) * 2

            endX = ((self.lastPose.x() / self.width) - 0.5) * 2
            endY = ((self.lastPose.y() / self.height) - 0.5) * 2

            # startPt = np.array([startX, startY, 0])
            # endPt = np.array([endX, endY, 0])
            # TO DO: Replace the 0s with the length of positions of where to draw w.r.t. the branch close by in the frustrum
            self.draw_lines.extend([startX, startY, 0, endX, endY, 0])

            print(f"Draw lines in mouseDraw {self.draw_lines}")
            self.drawLines()



    def drawLines(self):
        drawContext = QOpenGLContext()
        if not drawContext.create():
            raise Exception("Could not create a GL Context")
        # draw.aboutToBeDestroyed.connect(self.cleanUpGl)
        
        if not drawContext.makeCurrent(self):
            raise Exception("makeCurrent() failed (drawLines)")

        funcs = drawContext.functions()
        funcs.initializeOpenGLFunctions()

        # create the vao binder for drawing arrays
        vaoDrawBinder = QOpenGLVertexArrayObject.Binder(self.draw_vao)
        
        # Turn on the program the draws the lines
        self.draw_program.bind()

        # create and bind the vbo
        if not self.draw_vbo.create():
            print("Could not create a draw vbo array")
        if not self.draw_vbo.bind():
            print("Could not bind the draw vbo array")
        
        
        vertexLines = np.array(self.draw_lines, dtype=ctypes.c_float)
        print(vertexLines)
        # Allocate the vertex buffer object with the size of our vertices (in bytes)

        # Changed to draw_vertex
        self.draw_vbo.allocate(vertexLines.tobytes(), 
                              self.FLOATSIZE * vertexLines.size)


        funcs.glEnableVertexAttribArray(0)
        nullptr = VoidPtr(0)
        funcs.glVertexAttribPointer(0,                      # index
                                    3,                      # size (i.e., x, y, z)
                                    int(gl.GL_FLOAT),       # type
                                    int(gl.GL_FALSE),       # normalized?
                                    3 * self.FLOATSIZE,          # stride
                                    nullptr)                # pointer to starting position

        # draw the lines
        funcs.glDrawArrays(gl.GL_LINES,     # how to draw
                           0,                   # where to start drawing/location of values
                           int(vertexLines.size/3))                   # size

        
        # clean up the vbo and program
        self.draw_vbo.release()
        self.draw_vao.release()
        self.draw_program.release()  
        vaoDrawBinder = None
        
        drawContext.swapBuffers(self)
        drawContext.doneCurrent()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GLWidget()
    # window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())