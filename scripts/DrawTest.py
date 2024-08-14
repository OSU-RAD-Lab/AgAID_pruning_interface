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




# QtOpenGL.QGLWidget
# QOpenGLWidget
        

class Test(QOpenGLWidget):
    turnTableRotation = Signal(int)


    def __init__(self, parent=None):
        QOpenGLWidget.__init__(self, parent)

        # self.vertex = np.array([-0.5, -0.5, 0.0, 
        #                  0.5, -0.5, 0.0,
        #                  0.0, 0.5, 0.0], dtype=np.float32) # ctypes.c_float
        
        # self.vertices = np.array([[-0.5, -0.5, 0.0,],
        #                           [0.5, -0.5, 0.0],
        #                           [0.0, 0.5, 0.0]], dtype=np.float32)
        self.turntable = 0

        self.rotation = mt.create_identity()        # Gets an identity matrix
        self.projection = mt.create_identity()      # Sets the projection perspective
        self.view = mt.create_identity()            # sets the camera location
        
        self.projLoc = 0

        # self.mesh = Mesh("../obj_files/exemplarTree.obj")
        self.mesh = Mesh("../obj_files/testMonkey.obj")
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
        self.lightPos = [-1.0, 10.0, -1.0]
        self.lightColor = [1.0, 0.87, 0.13]
        self.camera_pos = [0, 0, 0]
        self.tree_color = [1.0, 1.0, 1.0, 1.0]
        self.triangle_color = [1.0, 0.0, 1.0, 0.0]
        
        # dimensions of the screen
        self.width = -1
        self.height = -1

        # for undoing drawings done by participants
        self.drawArray = {
            "vertices": [],
            "count": 0
        }

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle


    def setTurnTableRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.turntable:
            self.turntable = angle
            self.turnTableRotation.emit(angle)
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
    

    def initialize_skyBox(self):
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
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.skyVertices.nbytes, self.skyVertices, gl.GL_STATIC_DRAW)
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
        angle = self.angle_to_radians(self.turntable)
        rotation = mt.create_from_y_rotation(angle)
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

        


    def initializeGL(self):
        print(self.getGlInfo())
        # gl.glClearColor(0, 0.224, 0.435, 1)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        # gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        # gl.glClearColor(0.56, 0.835, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
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

        self.initialize_skyBox() # initialize all the skybox data


    def resizeGL(self, width, height):
        self.width = width 
        self.height = height

        side = min(width, height)
        if side < 0:
            return
        
        gl.glViewport(0, 0, width, height)
        # self.view = np.array(gl.glGetIntegerv(gl.GL_VIEWPORT))

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        # Set the perspective of the scene
        GLU.gluPerspective(45.0,    # FOV in y direction
                           aspect,  # FOV in x direction
                           0.1,     # z near
                           10.0)   # z far
        self.projection = np.array(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX))
        self.projection = np.transpose(self.projection)
        # print(self.projection)
        # self.projection = mt.create_perspective_projection_matrix(45.0, aspect, 0.1, 10.0)
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
        angle = self.angle_to_radians(self.turntable)
        x_rotation = mt.create_from_x_rotation(-np.pi/2) # for certain files to get upright
        rotation = mt.create_from_y_rotation(angle) # for rotating the modelView
        translation = np.transpose(mt.create_from_translation([0, 0, -5.883])) # 0, -2, -5.883
        # scale = mt.create_from_scale([-4.344, 4.1425, -9.99])
        scale = mt.create_from_scale([0.5, 0.5, 0.5])

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
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.vertices.size / 3)) 

        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()

        gl.glUseProgram(0)
        self.drawSkyBox()
        gl.glUseProgram(0)     


        
        # gl.glBindVertexArray(self.vao)
        # gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

       
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.vertex.size/3))

        # gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        
        # gl.glPopMatrix()

        # drawing for the lines we make
        # print(self.drawArray)

        # # FOR DRAWING THE LINES ADDED BY PARTICIPANTS
        # self.drawLines()
    
            
        # gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        # gl.glBindVertexArray(0)
        

    def loadCubeMap(self, directory):
       # Bind the texture and set our preferences of how to handle texture
        self.skyTexture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.skyTexture) 


        # Loop through the faces and add
        for type, face in zip(self.cubeTypes, self.cubeFaces):
            fname = str(directory) + str(face)
            # LOAD IN THE TEXTURE IMAGE AND READ IT IN TO OPENGL
            texImg = Image.open(fname)
            # texImg = texImg.transpose(Image.FLIP_TO_BOTTOM)
            texData = texImg.convert("RGB").tobytes()

            # need to load the texture into our program
            gl.glTexImage2D(type, 
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

        



    def mousePressEvent(self, event) -> None:
        self.startPose = QPoint(event.pos()) # returns the last position of the mouse when clicked
    


    def mouseReleaseEvent(self, event) -> None:
        self.lastPose = QPoint(event.pos())
        self.rayDraw()
        # _ = self.rayDirection(self.lastPose.x(), self.lastPose.y())

        # print("Ray: ", ray)
        # print("Last mouse position", self.lastPose)
        # self.mouseDraw() 
    

    def convertXYtoUV(self, x=0, y=0):
        u = ((2 * x) / self.width) - 1.0 
        v = 1.0 - ((2 * y) / self.height)
        return u, v

    def convertUVdtoXYZ(self, u=0, v=0, d=0):
        clip_space = [u, v, d, 1] # the equivalent of doing x'/w, y'/w, z'/w, w/w
        eye_space = np.linalg.inv(self.projection) @ np.transpose(clip_space) 
        world_space = np.linalg.inv(self.view) @ eye_space
        local_space = np.linalg.inv(self.model) @ world_space

        # converts back to x, y, z
        return local_space[:3] # only want the first 3 points


    
    ##############################################################################################
    # DESCRIPTION: Calculates the direction of a ray with the origin being at the camera position, 
    #              i.e., (0, 0, 0) but pointing in the direction of the mouse click (x, y)
    # INPUT:
    #   - x: an integer representing the x position on the screen
    #   - y: an integer representing the y position on the screen
    #
    # OUTPUT:
    #   - ray: a 1x4 array containing the direction of a ray based on the mouse click
    # See: https://antongerdelan.net/opengl/raycasting.html
    ##############################################################################################
    def rayDirection(self, x, y):
        # Use the x, y, z received from the mouse clicks
        # Set their range to [-1:1]

        # Convert x, y, to u, v (i.e., between [-1, 1])
        u, v = self.convertXYtoUV(x, y)
        print(f"Euclidean Coordinates: ({x}, {y})")
        print(f"UV Coordinates: ({u}, {v})")
        # u = ((2 * x) / self.width) - 1.0 
        # v = 1.0 - ((2 * y) / self.height)
        
        # Want the clip to be pointing down the direction of -z given that is where the camera is pointed
        clip = np.array([u, v, -1.0, 1.0])
        # originally eye
        eye = np.linalg.inv(self.projection) @ clip # convert to eye space from homogeneous clip space
        print("Eye Space Ray: ", np.transpose(eye))
        
        # ray inverse of our current view 
        # current view is down the -z access
        ray = np.linalg.inv(self.view) @ eye # convert to world view space
        # print("Ray (not normalized): ", np.transpose(ray))
        print(f"World Ray (not normalized): {np.transpose(ray)}")
        ray = ray / np.linalg.norm(ray)
        print(f"World Ray (normalized): {np.transpose(ray)}\n") 
        # print("Ray (normalized): ", np.transpose(ray))
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


    def rayDraw(self):
        # Use the first and last pose to get the midpoint 
        print(f"Start ({self.startPose.x()}, {self.startPose.y()})")
        print(f"End ({self.lastPose.x()}, {self.lastPose.y()})")
        midPt = [(self.startPose.x() + self.lastPose.x())/2, (self.startPose.y() + self.lastPose.y())/2]
        dir = self.rayDirection(x=midPt[0], y=midPt[1])[:3]
        
        # Determine faces in a given region



        # intercept = self.interception(origin=self.camera_pos, rayDirection=dir, vertexPos=self.vertexPos)
        # Need to pass in the vertices to the code
        # if len(intercept) == 0:
        #     print("No intercept detected")
        # return 


    def interception(self, origin, rayDirection, vertexPos):
    # Need to loop through the list of all vertex positions
    # grab out each face, and pass that to the rayCast algorithm
    # append intercepts to a list of values 
        intercepts = []
        for i in range(int(len(self.vertexPos) / 3)): # divide by 3 since 3 vertices per triangle face
            
            # Need to consider where the vertices are in world space!
            v1 = np.ones(4)
            v1[:3] = vertexPos[3*i]
            v1 = self.model @ np.transpose(v1)
            v1 = v1[:3]
            
            v2 = np.ones(4)
            v2[:3] = vertexPos[3*i + 1]
            v2 = self.model @ np.transpose(v2)
            v2 = v2[:3]

            v3 = np.ones(4)
            v3[:3] = vertexPos[3*i + 2]
            v3 = self.model @ np.transpose(v3)
            v3 = v3[:3]

            # Only want the first 3 values for the vertex points
            pt = self.rayCast(origin, rayDirection, v1, v2, v3)
            if pt is not None:
                intercepts.append(pt)
                print(f"Intercept at {pt}")
        
        return intercepts

    # return normalVector of the plane
    def normalVector(self, v1, v2, v3):
        return np.cross(v2 - v1, v3 - v1) 


    ###############################
    # Algorithm inspired by: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
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
        
        # DETERMINING IF THE RAY INTERXECTS USING BARYCENTRIC COORDINATES
        edgeBA = v2 - v1
        pEdgeA = pt - v1 # vector between possible intercept point and pt A of the triangle
        perp = np.cross(edgeBA, pEdgeA) # vector perpendicular to triangle's plane
        u = np.dot(normal, perp)
        if u < 0:
            return None

        edgeCB = v3 - v2
        pEdgeB = pt - v2
        perp = np.cross(edgeCB, pEdgeB)
        v = np.dot(normal, perp)
        if v < 0:
            return None
        
        edgeAC = v1 - v3
        pEdgeC = pt - v3
        perp = np.cross(edgeAC, pEdgeC)
        w = np.dot(normal, perp)
        if w < 0:
            return None

        return pt


    # The purpose of this button is to undo the drawing that participants have drawn so far on the screen
    def undoDraw(self):
        if self.drawArray["count"] > 0:
            self.drawArray["vertices"] = self.drawArray["vertices"][:-2]
            self.drawArray["count"] -= 2
            print("Removed line")
            self.update()               # need to redraw the scene so call update
        else:
            print("No line to remove")



    def rotatedDraw(self, x1, y1, x2, y2):
        # Calculate based on rotation
        diffAngle = self.angle_to_radians(90 - self.turntable)
        angle = self.angle_to_radians(self.turntable)
        # negativeAngle = self.angle_to_radians(360 - self.turntable)


        startX = x1 * np.cos(angle)
        startY = y1 
        # x1 * cos(90 - rotation) * sin(diffAngle)
        # At 90 degrees, z values are negated
        startZ = x1 * np.cos(diffAngle) # * np.sin(negativeAngle)

        # Calculate the new coordinates based on the rotation
        endX = x2 * np.cos(angle)
        endY = y2
        endZ = x2 * np.cos(diffAngle) # * np.sin(negativeAngle)

        if self.turntable >= 90:
            endZ *= -1
            startZ *= -1
        
        pt1 = np.array([startX, startY, startZ], dtype=np.float32)
        pt2 = np.array([endX, endY, endZ], dtype=np.float32)
        return pt1, pt2

    
    def mouseDraw(self):
        diffX = np.abs(self.startPose.x() - self.lastPose.x())
        diffY = np.abs(self.startPose.y() - self.lastPose.y())
        if diffX < (self.width / 20) and diffY < (self.height / 20):
            print(f"Difference between drawing points is too small {diffX} and {diffY}")
            # print(self.rayCast(self.startPose.x(), self.lastPose.y()))
        else:
            # Convert the coordinates between [-1, 1]
            x1 = ((2 * self.startPose.x()) / self.width) - 1.0
            y1 = 1.0 - ((2 * self.startPose.y() / self.height))
            print(f"Rotation: {self.turntable} pt 1: {x1},{y1}")

            # depth = gl.glReadPixels(x1, y1, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
            # print("Depth: ", depth)

            # model_view = np.array(gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX))
            # projection_view = np.array(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX))
            # view = np.array(gl.glGetIntegerv(gl.GL_VIEWPORT))

            # rotation = np.array(self.rotation)
            # point = GLU.gluUnProject(x1, y1, depth, model_view, projection_view, view)
            # print(point)
           
            x2 = ((2 * self.lastPose.x()) / self.width) - 1.0
            y2 = 1.0 - ((2 * self.lastPose.y() / self.height))

            pt1, pt2 = self.rotatedDraw(x1, y1, x2, y2)
            
            # startPt = self.projection @ self.rotation @ np.array([startX, startY, 0, 1])
            # print(f"Start pt {[startX, startY, 0]} vs translated point {startPt}")
            # print(self.rayCast(self.startPose.x(), self.lastPose.y()))

            self.drawArray["vertices"].append(pt1) # [x1, y1, 0]
            self.drawArray["vertices"].append(pt2) # [x2, y2, 0]
            self.drawArray["count"] += 2
            
            self.update()

    
    def drawLines(self):
        if self.drawArray["count"] > 0:
            gl.glLoadIdentity()
            gl.glPushMatrix()
            gl.glUseProgram(self.draw_program)
            # Set the uniforms
            # projLoc = gl.glGetUniformLocation(self.program, "projection")
            # viewLoc = gl.glGetUniformLocation(self.program, "view")
            rotationLoc = gl.glGetUniformLocation(self.draw_program, "rotation")
            gl.glUniformMatrix4fv(rotationLoc, 1, gl.GL_FALSE, self.rotation) # * self.camera

            projLoc = gl.glGetUniformLocation(self.draw_program, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_FALSE, self.projection)

            # viewLoc = gl.glGetUniformLocation(self.draw_program, "view")
            # gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_FALSE, self.view)

            # define information about the size of points
            gl.glPointSize(2)
            gl.glBegin(gl.GL_LINES)
            for v in self.drawArray["vertices"]:
                gl.glVertex3fv(v)
            # gl.glVertex3d(x, y, 0)
            gl.glEnd()
            gl.glPopMatrix()



# QWidget 
# QtOpenGL.QMainWindow 
class Window(QMainWindow):

    def __init__(self, parent=None):
        # super(Window, self).__init__()
        QMainWindow.__init__(self, parent)
        # QtOpenGL.QMainWindow.__init__(self)
        self.resize(700, 700)
        self.setWindowTitle("TEST Window")

        self.central_widget = QWidget() # GLWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        
        self.glWidget = Test()
        self.layout.addWidget(self.glWidget)
        
        # self.glWidget = Test()
        # self.setCentralWidget(self.glWidget)

        # self.layout = QGridLayout(self.central_widget)
        
        self.slider = self.createSlider()
        self.slider.valueChanged.connect(self.glWidget.setTurnTableRotation)
        self.glWidget.turnTableRotation.connect(self.slider.setValue)

        self.layout.addWidget(self.slider)


        self.undoButton = QPushButton("Undo Button")
        self.undoButton.clicked.connect(self.glWidget.undoDraw)
        # self.layout.addWidget(self.undoButton)

        # main_layout = QVBoxLayout()
        # main_layout.addWidget(self.glWidget)
        # main_layout.addWidget(self.slider)
        # self.setLayout(main_layout)


    
    
    def createSlider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 360) # 0 - 360*16
        slider.setSingleStep(1) # 
        slider.setPageStep(10)
        slider.setTickPosition(QSlider.TicksBelow)

        return slider
        




if __name__ == '__main__':

    app = QApplication(sys.argv)

    # fmt = QSurfaceFormat()
    # fmt.setDepthBufferSize(24)
    # # print(QCoreApplication.arguments())
    # QSurfaceFormat.setDefaultFormat(fmt)


    window = Window() # GLDemo()
    # window = MainWindow()
    window.show()
    sys.exit(app.exec_())