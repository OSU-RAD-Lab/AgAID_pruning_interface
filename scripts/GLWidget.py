#!/usr/bin/env python3

#!/usr/bin/env python3
import sys
sys.path.append('../')

import os
os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"


from PySide2 import QtCore, QtGui, QtOpenGL
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint, QCoreApplication
from PySide2.QtOpenGL import QGLWidget, QGLContext
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4, QSurfaceFormat
from shiboken2 import VoidPtr
from OpenGL.GL.shaders import compileShader, compileProgram


from scripts.MeshAndShaders import Mesh, Shader

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
import numpy as np
import ctypes                 # to communicate with c code under the hood
import pyrr.matrix44 as mt
from PIL import Image




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

    turnTableRotation = Signal(int)
    verticalRotation = Signal(int)


    def __init__(self, parent=None, wholeView=False):
        QOpenGLWidget.__init__(self, parent)

        # self.vertex = np.array([-0.5, -0.5, 0.0, 
        #                  0.5, -0.5, 0.0,
        #                  0.0, 0.5, 0.0], dtype=np.float32) # ctypes.c_float
        
        # self.vertices = np.array([[-0.5, -0.5, 0.0,],
        #                           [0.5, -0.5, 0.0],
        #                           [0.0, 0.5, 0.0]], dtype=np.float32)
        self.wholeView = wholeView
        self.turntable = 0
        self.vertical = 0
        self.rotation = mt.create_identity()        # Gets an identity matrix
        self.projection = mt.create_identity()      # Sets the projection perspective
        self.view = mt.create_identity()            # sets the camera location
        self.projLoc = 0

        self.ZNEAR = 0.1
        self.ZFAR = 10.0

        # self.mesh = Mesh("../obj_files/exemplarTree.obj")
        self.mesh = Mesh("../obj_files/testMonkey.obj")
        # self.mesh = Mesh("../obj_files/textureTree.obj")
        # self.mesh = Mesh("../obj_files/skyBox.obj")
        self.vertices = np.array(self.mesh.vertices, dtype=np.float32) # contains texture coordinates, vertex normals, vertices      
        self.texture = None
        self.vao = None
        self.vbo = None 

        # GETTING THE BACKGROUND SKY BOX
        self.skyMesh = Mesh("../obj_files/skyBox.obj")
        self.skyVertices = np.array(self.skyMesh.vertices, dtype=np.float32)
        self.skyProgram = None
        self.skyTexture = None
        self.skyVAO = None
        self.skyVBO = None
        self.skyMapDir = "../textures/skymap/"        
        self.cubeFaces = ["px.png", # right
                          "nx.png", # left
                          "py.png", # top
                          "ny.png", # bottom
                          "pz.png", # front
                          "nz.png"] # back
        
        

        # UNIFORM VALUES FOR SHADERS
        self.lightPos = [0, 5.0, 0] # -1, 5.0, -1
        self.lightColor = [1.0, 1.0, 0] # 1.0, 0.87, 0.13
        self.camera_pos = [0, 0, 0]
        self.tree_color = [1.0, 1.0, 1.0, 1.0]
        self.triangle_color = [1.0, 0.0, 1.0, 0.0]
        
        # dimensions of the screen
        self.width = -1
        self.height = -1

        # DRAWING VALUES
        self.drawVAO = None
        self.drawVBO = None
        self.drawProgram = None
        self.drawLines = False # dtermine if to show the line
        self.drawVertices = np.zeros(3000, dtype=np.float32) # give me a set of values to declare for vbo
        self.drawCount = 0
        # self.drawVertices[:18] = np.array([-2, -2, 0, # -2, -2
        #                               2, -2, 0,
        #                               2, -2, -4.99,
        #                               -2, -2, -4.99,
        #                               -3.99, -3.99, -4.99,
        #                               3.99, -3.99, -4.99], dtype=np.float32)
        


        self.SIMPLE_VERTEX_SHADER = """
        # version 330 core

        layout (location = 0) in vec3 aPos;

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        out vec4 color;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            color = vec4(1.0, 0.0, 0.0, 0.0);
        }
        """

        self.SIMPLE_FRAGMENT_SHADER = """
        # version 330 core
        in vec4 color;
        out vec4 FragColor;

        void main()
        {
            FragColor = color;
        }
        """

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle


    def setTurnTableRotation(self, angle):
        # angle = self.normalizeAngle(angle)
        if angle != self.turntable:
            self.turntable = angle
            self.turnTableRotation.emit(angle)
            self.update()

    
    def setVerticalRotation(self, angle):
        # angle = self.normalizeAngle(angle)
        if angle != self.vertical:
            self.vertical = angle
            self.verticalRotation.emit(angle)
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
    

    def initializeSkyBox(self):
        gl.glUseProgram(0)
        vertexShader = Shader("vertex", "skybox_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "skybox_shader.frag").shader

        self.skyProgram = gl.glCreateProgram()
        gl.glAttachShader(self.skyProgram, vertexShader)
        gl.glAttachShader(self.skyProgram, fragmentShader)
        gl.glLinkProgram(self.skyProgram)

        gl.glUseProgram(self.skyProgram)
        # create and bind the vertex attribute array
        self.skyVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.skyVAO)

        # bind the vertex buffer object
        self.skyVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.skyVBO)

        # laod the texture cube map
        self.skyTexture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.skyTexture) 


        # Loop through the faces and add
        for i, face in enumerate(self.cubeFaces):
            fname = str(self.skyMapDir) + str(face)
            # LOAD IN THE TEXTURE IMAGE AND READ IT IN TO OPENGL
            texImg = Image.open(fname)
            # texImg = texImg.transpose(Image.FLIP_TO_BOTTOM)
            texData = texImg.convert("RGB").tobytes()

            # need to load the texture into our program
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
                            0,                      # mipmap setting (can change for manual setting)
                            gl.GL_RGB,              # texture file storage format
                            texImg.width,           # texture image width
                            texImg.height,          # texture image height
                            0,                      # 0 for legacy 
                            gl.GL_RGB,              # format of the texture image
                            gl.GL_UNSIGNED_BYTE,    # datatype of the texture image
                            texData)                # data texture  
        
        # SET THE PARAMETERS OF THE TEXTURE
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR) 
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        # Reshape the list for loading into the vbo 
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.skyVertices.nbytes, self.skyVertices, gl.GL_DYNAMIC_DRAW) # gl.GL_STATIC_DRAW
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex.nbytes, self.vertex, gl.GL_STATIC_DRAW)

        # SET THE ATTRIBUTE POINTERS SO IT KNOWS LCOATIONS FOR THE SHADER
        stride = self.skyVertices.itemsize * 8 # times 8 given there are 8 vertices across texture coords, normals, and vertex positions before next set
        # TEXTURE COORDINATES
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)) 

        # NORMAL VECTORS (normals)
        gl.glEnableVertexAttribArray(1)  # SETTING THE NORMAL
        # gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(0)) # for location = 0
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8)) # starts at position 2 with size 4 for float32 --> 2*4

        # VERTEX POSITIONS (vertexPos)
        gl.glEnableVertexAttribArray(2) # SETTING THE VERTEX POSITIONS
        # gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(12)) # for location = 1
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(20)) # starts at position 5 with size 4 --> 5*4


    def drawSkyBox(self):
        gl.glLoadIdentity()
        gl.glPushMatrix()
        gl.glUseProgram(0)
        # gl.glDisable(gl.GL_CULL_FACE)
       
        # gl.glDisable(gl.GL_DEPTH_TEST)
        # set the depth function to equal
        oldDepthFunc = gl.glGetIntegerv(gl.GL_DEPTH_FUNC)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_FALSE)
        

        # Deal with the rotation of the object
        # scale = mt.create_from_scale([-4.3444, 4.1425, -10.00])
        scale = mt.create_from_scale([-4, 4, -9.99])
        # translate = mt.create_from_translation([0, -4, 0])
        hAngle = self.angle_to_radians(self.turntable)
        vAngle = self.angle_to_radians(self.vertical)

        rotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)
        model = rotation @ scale # for rotating the modelView

        # set last row and column to 0 so it doesn't affect translations but only rotations
        view = mt.create_identity()
        view[:,3] = np.zeros(4)
        view[3, :] = np.zeros(4)

        # set the shader for the skybox
        gl.glUseProgram(self.skyProgram)

        modelLoc = gl.glGetUniformLocation(self.skyProgram, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation

        projLoc = gl.glGetUniformLocation(self.skyProgram, "projection")
        gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection) # use the same projection values

        viewLoc = gl.glGetUniformLocation(self.skyProgram, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, view) # use the same location view values

        # BIND VAO AND TEXTURE
        gl.glBindVertexArray(self.skyVAO)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.skyTexture)

        gl.glDrawArrays(gl.GL_QUADS, 0, int(self.skyVertices.size)) 
        # gl.glDrawElements(gl.GL_QUADS, int(self.skyVertices.size), gl.GL_UNSIGNED_INT, 0)

        # Reset values
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(oldDepthFunc) # set back to default value
        # gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glUseProgram(0)
        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()



    def initializeDrawing(self):
        gl.glUseProgram(0)

        vertexShader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vertexShader, self.SIMPLE_VERTEX_SHADER)
        gl.glCompileShader(vertexShader)
    
        fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fragmentShader, self.SIMPLE_FRAGMENT_SHADER)
        gl.glCompileShader(fragmentShader)

        self.drawProgram = gl.glCreateProgram()
        gl.glAttachShader(self.drawProgram, vertexShader)
        gl.glAttachShader(self.drawProgram, fragmentShader)
        gl.glLinkProgram(self.drawProgram)

        gl.glUseProgram(self.drawProgram)

        self.drawVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.drawVAO)

        self.drawVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.drawVBO)

        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.drawVertices.nbytes, self.drawVertices, gl.GL_STATIC_DRAW)

        stride = self.drawVertices.itemsize * 3

        # enable the pointer for the shader
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))  

    
    def drawPruningLines(self):
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        gl.glUseProgram(self.drawProgram) 
        
        # SET THE LIGHT COLOR
        # angle = self.angle_to_radians(self.turntable)
        # # x_rotation = mt.create_from_x_rotation(-np.pi/2) # for certain files to get upright
        # model = mt.create_from_y_rotation(angle) # for rotating the modelView
        
        modelLoc = gl.glGetUniformLocation(self.drawProgram, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, self.model) # self.rotation

        projLoc = gl.glGetUniformLocation(self.drawProgram, "projection")
        gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)

        viewLoc = gl.glGetUniformLocation(self.drawProgram, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

        # # self.draw(self.vertices)
        # end = self.drawCount * 3
        gl.glBindVertexArray(self.drawVAO)
        # gl.glLineWidth(2.0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.drawVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.drawVertices.nbytes, self.drawVertices, gl.GL_DYNAMIC_DRAW)

        # gl.glPointSize(3.0)
        gl.glLineWidth(2.0)
        gl.glDrawArrays(gl.GL_QUADS, 0, int(self.drawVertices.size / 3))
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.drawCount) 

        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0)
        


    def initializeGL(self):
        print(self.getGlInfo())
        # gl.glClearColor(0, 0.224, 0.435, 1)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        # gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        # gl.glClearColor(0.56, 0.835, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        
        # TREE SHADER
        vertexShader = Shader("vertex", "shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "shader.frag").shader

        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vertexShader)
        gl.glAttachShader(self.program, fragmentShader)
        gl.glLinkProgram(self.program)

        gl.glUseProgram(self.program)
        # print(gl.glGetError())
       
        # create and bind the vertex attribute array
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        # bind the vertex buffer object
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # Bind the texture and set our preferences of how to handle texture
        self.texture = gl.glGenTextures(1)
        # print(f"texture ID {self.texture}")
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture) 
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT) # will repeat the image if any texture coordinates are outside [0, 1] range for x, y values
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST) # describing how to perform texture filtering (could use a MipMap)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # LOAD IN THE TEXTURE IMAGE AND READ IT IN TO OPENGL
        texImg = Image.open("../textures/bark.jpg")
        # texImg = Image.open("../textures/testTexture.png")
        # texImg = texImg.transpose(Image.FLIP_TO_BOTTOM)
        texData = texImg.convert("RGB").tobytes()

        # need to load the texture into our program
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 
                        0,                      # mipmap setting (can change for manual setting)
                        gl.GL_RGB,              # texture file storage format
                        texImg.width,           # texture image width
                        texImg.height,          # texture image height
                        0,                      # 0 for legacy 
                        gl.GL_RGB,              # format of the texture image
                        gl.GL_UNSIGNED_BYTE,    # datatype of the texture image
                        texData)                # data texture 


        # Reshape the list for loading into the vbo 
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex.nbytes, self.vertex, gl.GL_STATIC_DRAW)

        # SET THE ATTRIBUTE POINTERS SO IT KNOWS LCOATIONS FOR THE SHADER
        stride = self.vertices.itemsize * 8 # times 8 given there are 8 vertices across texture coords, normals, and vertex positions before next set
        # TEXTURE COORDINATES
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)) 

        # NORMAL VECTORS (normals)
        gl.glEnableVertexAttribArray(1)     # SETTING THE NORMAL
        # gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(0)) # for location = 0
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8)) # starts at position 2 with size 4 for float32 --> 2*4

        # VERTEX POSITIONS (vertexPos)
        gl.glEnableVertexAttribArray(2) # SETTING THE VERTEX POSITIONS
        # gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(12)) # for location = 1
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(20)) # starts at position 5 with size 4 --> 5*4


        # INITIALIZE THE DRAWING PROGRAM
        self.initializeDrawing() # initialize the places for the drawing of values

        self.initializeSkyBox() # initialize all the skybox data
        
        
    def resizeGL(self, width, height):
        self.width = width 
        self.height = height

        side = min(width, height)
        if side < 0:
            return
        
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        # Set the perspective of the scene
        GLU.gluPerspective(45.0,            # FOV in y direction
                           aspect,          # FOV in x direction
                           self.ZNEAR,      # z near
                           self.ZFAR)       # z far
        self.projection = np.array(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX))
        self.projection = np.transpose(self.projection)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    
    def angle_to_radians(self, angle):
        return angle * (np.pi / 180.0)

    
    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glLoadIdentity()
        gl.glPushMatrix()

        # set the perspective
        # PUT THE OBJECT IN THE CORRECT POSITION ON THE SCREEN WITH 0, 0, 0 BEING THE CENTER OF THE TREE
        # Specifically putting the object at x = 0, 0, -5.883 --> u = 0, v = 0, d = 0.5
        # Based on calculations from the projection matrix with fovy = 45, aspect = (646/616), near = 0.1, far = 10.0
        # rotate and translate the model to the correct position
        # Deal with the rotation of the object
        hAngle = self.angle_to_radians(self.turntable)
        vAngle = self.angle_to_radians(self.vertical)

        rotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

        if self.wholeView: # looking at the whole tree view
            translation = np.transpose(mt.create_from_translation([0, 0, -5.0])) # 0, -0, -5.883
        else:
            translation = np.transpose(mt.create_from_translation([0, 0, -2.0]))
        # scale = mt.create_from_scale([-4.344, 4.1425, -9.99])
        scale = mt.create_from_scale([1, 1, 1])

        self.model = mt.create_identity()
        self.model = translation @ rotation @ scale # @ x_rotation  # x-rotation only needed for some files
        # self.model = rotation @ scale

        # CALCULATE NEW VIEW (at our origin (0, 0, 0))
        self.view = mt.create_identity() # want to keep the camera at the origin (0, 0, 0)   

        # SET THE UNIFORMS FOR THE PROJECTION * CAMERA MOVEMENTS
        # projLoc = gl.glGetUniformLocation(self.program, "projection")
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glUseProgram(self.program) 
        

        # SET THE LIGHT COLOR
        lightPosLoc = gl.glGetUniformLocation(self.program, "lightPos")
        gl.glUniform3fv(lightPosLoc, 1, self.lightPos)

        lightColorLoc = gl.glGetUniformLocation(self.program, "lightColor")
        gl.glUniform3fv(lightColorLoc, 1, self.lightColor)

        modelLoc = gl.glGetUniformLocation(self.program, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, self.model) # self.rotation

        projLoc = gl.glGetUniformLocation(self.program, "projection")
        gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)

        viewLoc = gl.glGetUniformLocation(self.program, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

        # # self.draw(self.vertices)
        gl.glBindVertexArray(self.vao)
        gl.glPointSize(2.0)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.vertices.size / 3)) 

        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()


        # WANT TO DRAW THE POINTS BASED ON WHAT SOMEONE WAS DRAWING
        if self.drawLines:
            self.drawPruningLines()

        
        # TO ADD CONCEPT TO DRAW BOUNDING BOX FOR WHOLE VIEW CAMERA

        # TO ADD CONCEPT FOR DRAWING HINTS/CORRECT PRUNING CUTS WHEN ASKED

        # DRAW THE BACKGROUND SKYBOX
        gl.glUseProgram(0)
        self.drawSkyBox()
        gl.glUseProgram(0) 
    

    # MOUSE ACTION EVENTS WITH GLWIDGET SCREEN
    def mousePressEvent(self, event) -> None:
        self.startPose = QPoint(event.pos()) # returns the last position of the mouse when clicked
    

    def mouseReleaseEvent(self, event) -> None:
        self.lastPose = QPoint(event.pos())
        self.rayDraw()
        # _ = self.rayDirection(self.lastPose.x(), self.lastPose.y())
    

    def convertXYtoUV(self, x=0, y=0):
        u = ((2 * x) / self.width) - 1.0 
        v = 1.0 - ((2 * y) / self.height)
        return u, v


    def convertUVDtoXYZ(self, u=0, v=0, d=0):
        clip_space = np.array([u, v, d, 1]) # the equivalent of doing x'/w, y'/w, z'/w, w/w
        # print(f"Clip space: ", clip_space)
        # originally eye
        eye = np.linalg.inv(self.projection) @ np.transpose(clip_space) # convert to eye space from homogeneous clip space
        # print("Eye Space Ray: ", np.transpose(eye))
        
        # ray inverse of our current view 
        # current view is down the -z access
        world_space = np.linalg.inv(self.view) @ eye # convert to world view space

        # convert back to local space
        local_space = np.linalg.inv(self.model) @ world_space
        
        # converts back to x, y, z
        return local_space # only want the first 3 points



    def convertWorldtoUVD(self, pt):
        vertex = np.ones(4)
        vertex[:3] = pt

        mvp = self.projection @ self.view @ np.transpose(vertex)
        mvp /= mvp[3]

        return mvp

    def convertUVDtoWorld(self, pt):
        clip = np.ones(4)
        clip[:3] = pt
        eye = np.linalg.inv(self.projection) @ clip
        world = np.linalg.inv(self.view) @ eye
        world /= world[3]
        return world[:3]


    
    ##############################################################################################
    # DESCRIPTION: Calculates the direction of a ray with the origin being at the camera position, 
    #              i.e., (0, 0, 0) but pointing in the direction of the mouse click (x, y)
    # INPUT:
    #   - x: an integer representing the x position on the screen
    #   - y: an integer representing the y position on the screen
    #
    # OUTPUT:
    #   - ray: a 1x4 array containing the direction of a ray based on the mouse click in world space
    # See: https://antongerdelan.net/opengl/raycasting.html
    ##############################################################################################
    def rayDirection(self, x, y):
        # Use the x, y, z received from the mouse clicks
        # Set their range to [-1:1]

        # Convert x, y, to u, v (i.e., between [-1, 1])
        u, v = self.convertXYtoUV(x, y)
        # print(f"Euclidean Coordinates: ({x}, {y})")
        # print(f"UV Coordinates: ({u}, {v})")
        
        # Want the clip to be pointing down the direction of -z given that is where the camera is pointed
        clip = np.array([u, v, -1.0, 1.0])
        # originally eye
        eye = np.linalg.inv(self.projection) @ clip # convert to eye space from homogeneous clip space
        # print("Ray direction (eye): ", eye)
        # print("Eye Space Ray: ", np.transpose(eye))

        ray_eye = np.array([eye[0], eye[1], -1.0, 0.0])
        
        # ray inverse of our current view 
        # current view is down the -z access
        ray = np.linalg.inv(self.view) @ ray_eye # convert to world view space
        # print("Ray (not normalized): ", np.transpose(ray))
        # print(f"World Ray (not normalized): {np.transpose(ray)}")
        ray = ray / np.linalg.norm(ray)
        # print(f"World Ray (normalized): {np.transpose(ray)}\n") 
        # print("Ray (normalized): ", np.transpose(ray))
        return ray


    def convertXYToWorld(self, x, y):
        u, v = self.convertXYtoUV(x, y)

        # Want the clip to be pointing down the direction of -z given that is where the camera is pointed
        clip = np.array([u, v, 0, 1.0])
        # originally eye
        eye = np.linalg.inv(self.projection) @ clip # convert to eye space from homogeneous clip space
        # print("Eye Space Ray: ", np.transpose(eye))

        ray_eye = np.array([eye[0], eye[1], 0, 0.0])
        
        # ray inverse of our current view 
        # current view is down the -z access
        ray = np.linalg.inv(self.view) @ ray_eye # convert to world view space
        return ray


    def convertToUVD(self, vertex):
        # extract all the vertices of the array
        # perform the projection * view * matrix * vertex multiplication
        position = np.ones(4)
        position[:3] = vertex

        modelVertex = self.model @ np.transpose(position)
        modelViewVertex = self.view @ modelVertex
        mvpVertex = self.projection @ modelViewVertex

        mvpVertex = mvpVertex / mvpVertex[3] # divide by w to normalize

        return mvpVertex[:3]


    def convertWorldToLocal(self, pt):
        vertex = np.ones(4)
        vertex[:3] = pt
        # print(vertex)
        local = np.linalg.inv(self.model) @ np.transpose(vertex)
        # print(local)
        return local[:3]


    def convertToWorld(self, pt):
        vertex = np.ones(4)
        vertex[:3] = pt
        world = self.model @ np.transpose(vertex)
        return world[:3]


    def addDrawVertices(self, pt1, pt2, pt3, pt4):
        quad = [pt1, pt2, pt3, pt4]

        for i in range(4):
            # start adding at point 3*count in draw array
            start = (self.drawCount+i) * 3
            # localPt = self.convertWorldToLocal(quad[i])
            self.drawVertices[start:start+3] = quad[i]
            # self.draw["vertices"].extend(quad[i])
        self.drawCount += 4 # add 4 since added 4 vertices to create a quad
        # self.draw["count"] += 4


    def get_drawn_coords(self, u, v, z):
        # convert z to d
        depth = self.convertToUVD([0, 0, z])
        print(f"UVD pt: {u}, {v}, {depth[2]}")
        localPt = self.convertUVDtoXYZ(u=u, v=v, d=depth[2])            
        # need to divide by w value to get x, y, z
        localPt /= localPt[-1]
        return localPt[:3]


    def rayDraw(self):
        # Use the first and last pose to get the midpoint 
        print(f"Start ({self.startPose.x()}, {self.startPose.y()})")
        print(f"End ({self.lastPose.x()}, {self.lastPose.y()})")

        # checking the midpoint value
        # midPt = [(self.startPose.x() + self.lastPose.x())/2, (self.startPose.y() + self.lastPose.y())/2]
        # dir = self.rayDirection(x=midPt[0], y=midPt[1])[:3]
        # print(dir)

        u1, v1 = self.convertXYtoUV(x=self.startPose.x(), y=self.startPose.y())
        u2, v2 = self.convertXYtoUV(x=self.lastPose.x(), y=self.lastPose.y())

        # get the set of faces that could possibly intersect the line
        # intersectFaces = self.mesh.faces_in_area(pt1, pt2, self.model)
        # print("Intersect Faces: ", int(intersectFaces.size / 9))

        # returns intersect faces in local coordinates
        intersectFaces = self.mesh.intersect_faces(u1=u1, v1=v1, u2=u2, v2=v2, proj=self.projection, view=self.view, model=self.model)
        
        if intersectFaces is not None:
            # Determine faces in a given region   
            dirPt = [(self.startPose.x() + self.lastPose.x())/2, (self.startPose.y() + self.lastPose.y())/2]
            dir = self.rayDirection(x=dirPt[0], y=dirPt[1])[:3]
            depth, intercept = self.interception(origin=self.camera_pos, rayDirection=dir, faces=intersectFaces)
            # Need to pass in the vertices to the code
            if len(intercept) == 0:
                print("No intercept detected")
            else:
                # Now I need to find the min and max z but max their value slightly larger for the rectangle
                # Depth given in world space
                self.drawLines = True
                minZ = np.min(depth)
                maxZ = np.max(depth)
                print(f"Local Zs: {minZ} & {maxZ}")
                # testLocal = self.eyeCast(self.startPose.x(), self.startPose.y(), minZ)

                # TROUBLE SECTION!!!!!
                drawPt1 = self.get_drawn_coords(u1, v1, minZ)
                drawPt2 = self.get_drawn_coords(u1, v1, maxZ)
                drawPt3 = self.get_drawn_coords(u2, v2, maxZ)
                drawPt4 = self.get_drawn_coords(u2, v2, minZ)

                print(drawPt1)
                print(drawPt2)
                print(drawPt3)
                print(drawPt4)
                # print(f"New coordinates at {self.startPose.x()},{self.startPose.y()},{minZ} is {drawPt1}")
                # print(f"New coordinates at {self.startPose.x()},{self.startPose.y()},{maxZ} is {drawPt2}")
                # print(f"New coordinates at {self.lastPose.x()},{self.lastPose.y()},{maxZ} is {drawPt3}")
                # print(f"New coordinates at {self.lastPose.x()},{self.lastPose.y()},{minZ} is {drawPt4}")

                self.addDrawVertices(drawPt1, drawPt2, drawPt3, drawPt4)

                # UPDATE VBO TO INCORPORATE THE NEW VERTICES
                gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)

        self.update()              
                             

    ###############################################
    # INPUT: 
    #   - origin: a 3x1 vertex representing the origin position (0, 0, 0)
    #   - rayDirection: a vector representing the direction the ray is travelling towards the faces
    #   - faces: an array of faces represented by 3 vertices in world coordinates that exist within the drawing coordinates
    # OUTPUT:
    #   - an array of 3x1 vertices (in world coordinates) that intersect a face given a ray going in direction rayDirection starting at the origin
    ################################################
    def interception(self, origin, rayDirection, faces):
    # Need to loop through the list of all vertex positions
    # grab out each face, and pass that to the rayCast algorithm
    # append intercepts to a list of values 
        depth = []
        intercepts = []
        for face in faces:
            # v1 = self.convertUVDtoWorld(face[0])
            # v2 = self.convertUVDtoWorld(face[1])
            # v3 = self.convertUVDtoWorld(face[2])

            v1 = self.convertToWorld(face[0])
            v2 = self.convertToWorld(face[1])
            v3 = self.convertToWorld(face[2])
            # Only want the first 3 values for the vertex points
            
            pt = self.rayCast(origin, rayDirection, v1, v2, v3)
            if pt is not None:
                intercepts.append(pt)
                depth.append(face[2]) # append the local depth to the depth
                # print(f"Pt: {pt}")
                # print(f"Intercepted face: {face}")

                # # extend the drawVertices so we can see what faces are intersected by the ray
                # wv1 = np.ones(4)
                # wv1[:3] = v1
                # local1 = np.linalg.inv(self.model) @ np.transpose(wv1)
                # self.drawVertices[start:start+3] = local1[:3]

                # wv2 = np.ones(4)
                # wv2[:3] = v2
                # local2 = np.linalg.inv(self.model) @ np.transpose(wv2)
                
                # # self.drawVertices[start:start+3] = local2[:3]
                # self.drawVertices[start] = local2[0]
                # self.drawVertices[start+1] = local2[1]
                # self.drawVertices[start+2] = local2[2]

                # start = start + 3
                # wv3 = np.ones(4)
                # wv3[:3] = v3
                # local3 = np.linalg.inv(self.model) @ np.transpose(wv3)
                # self.drawVertices[start] = local3[0]
                # self.drawVertices[start+1] = local3[1]
                # self.drawVertices[start+2] = local3[2]
                # self.drawCount += 3
                # self.drawLines = True

                
                # print(f"Intercept at {pt}")
        return depth, intercepts



    # return normalVector of the plane
    def normalVector(self, v1, v2, v3):
        return np.cross(v2 - v1, v3 - v1) 


    ###############################
    # Algorithm inspired by: 
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
    ################################
    def rayCast(self, origin, rayDirection, v1, v2, v3):
        # take a 3D direction and calculate if it intersects the plane
        normal = self.normalVector(v1, v2, v3)
        # area = np.linalg.norm(normal)
        # print(normal)

        denominator = np.dot(rayDirection, normal)    
        if denominator <= 1e-3: # close to 0 emaning it is almost parallel
            return None # no intersect due to plane and ray being parallel
        
        # dist = origin - plane[0] # change from A to center point of the plane
        dist = np.dot(-normal, v1)
        numerator = -(np.dot(normal, origin) + dist)
        # numerator = np.dot(origin, normal) + plane[0] # to the distance of pt1 on the plane
        t = numerator / denominator
        if t < 0:
            return None # triangle is behind the ray
        
        pt = origin + t * rayDirection # possible intercept point
        
        # DETERMINING IF THE RAY INTERSECTS USING BARYCENTRIC COORDINATES WITHIN SOME THRESHOLD
        delta = 0.1 # arrived after testing as most values were negative -0.004 which is close enough to zero
        
        edgeBA = v2 - v1
        pEdgeA = pt - v1 # vector between possible intercept point and pt A of the triangle
        perp = np.cross(edgeBA, pEdgeA) # vector perpendicular to triangle's plane
        u = np.dot(normal, perp)
        if abs(u) > delta:
            return None
        

        edgeCB = v3 - v2
        pEdgeB = pt - v2
        perp = np.cross(edgeCB, pEdgeB)
        v = np.dot(normal, perp)
        if abs(v) > delta:
            return None
        
        edgeAC = v1 - v3
        pEdgeC = pt - v3
        perp = np.cross(edgeAC, pEdgeC)
        w = np.dot(normal, perp)
        if abs(w) > delta:
            return None

        
        return pt


    # The purpose of this button is to undo the drawing that participants have drawn so far on the screen
    def undoDraw(self):
        if self.drawCount > 0:
            start = (self.drawCount - 4) * 3
            end = self.drawCount * 3

            print(f"Delete vertices from indices {start}:{end}")
            # need to replace all the values from drawCount
            self.drawVertices[start:end] = np.zeros(12)
            self.drawCount -= 4
            gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)
            self.update()
        else:
            print("No line to remove")
        # if self.drawArray["count"] > 0:
        #     self.drawArray["vertices"] = self.drawArray["vertices"][:-2]
        #     self.drawArray["count"] -= 2
        #     print("Removed line")
        #     self.update()               # need to redraw the scene so call update
        # else:
        #     print("No line to remove")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GLWidget()
    # window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())