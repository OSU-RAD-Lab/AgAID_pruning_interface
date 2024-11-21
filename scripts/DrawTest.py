#!/usr/bin/env python3
import sys
sys.path.append('../')

import os
os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"


from PySide2 import QtCore, QtGui, QtOpenGL
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget, QComboBox
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint, QCoreApplication, QPoint
from PySide2.QtOpenGL import QGLWidget, QGLContext
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4, QSurfaceFormat, QPainter, QFont
from shiboken2 import VoidPtr
from OpenGL.GL.shaders import compileShader, compileProgram


from scripts.MeshAndShaders import Mesh, Shader

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
import numpy as np
import ctypes                 # to communicate with c code under the hood
import pyrr.matrix44 as mt
from PIL import Image
import time
import math
from freetype import * # allows us to display text on the screen
from JSONFileReader import *




# QtOpenGL.QGLWidget
# QOpenGLWidget

class Character:
    def __init__(self, texID, size, bearing, advance):
        self.texID = texID
        self.size = size  # a tuple storing the size of the glyph
        self.bearing = bearing # a tuple describing the bearing of the glyph
        self.advance = advance


class Label:
    def __init__(self, texID, vao, vbo, world):
        self.texID = texID
        self.vao = vao
        self.vbo = vbo
        self.world = world # matrix of where to put the object


class Shapes:
    def __init__(self, shape="square"):
        if shape == "square": # Square
            self.vertices = np.array([-1, -1, 0,  
                                    -1, 1, 0,
                                    1, 1, 0,
                                    1, -1, 0,
                                    -1, -1, 0], dtype=np.float32)

        if shape == "circle": # Circle
            self.vertices = np.array([0, 1, 0,
                                      0.5, 0.866, 0,
                                      0.866, 0.5, 0,
                                      1, 0, 0, 
                                      0.866, -0.5, 0,
                                      0.5, -0.866, 0,
                                      0, -1, 0,
                                      -0.5, -0.866, 0,
                                      -0.866, -0.5, 0,
                                      -1, 0, 0, 
                                      -0.866, 0.5, 0, 
                                      -0.5, 0.866, 0, 
                                      0, 1, 0  
                                    ], dtype=np.float32)
            self.vertices *= 0.1 # make it even smaller


class Test(QOpenGLWidget):
    turnTableRotation = Signal(int)
    verticalRotation = Signal(int)
    manipulationIndex = Signal(int)


    def __init__(self, parent=None, wholeView=False, meshDictionary=None, fname=None, jsonData=None, manipulation={}, screenType="normal"):
        QOpenGLWidget.__init__(self, parent)

        self.jsonData = jsonData
        self.toManipulate = False
        self.toBin = True
        self.manipulation = manipulation
        self.screenType = screenType
        # self.toManipulate = manipulation["Display"] # Either True or False
        # self.manipulationJSON = manipulation["JSON"] # the json data
        self.fname = str(fname)
        # self.manipulation = manipulation["Directory"]
        self.wholeView = wholeView # whether to show the whole view
        self.turntable = 0
        self.vertical = 0
        self.index = 0
        self.rotation = mt.create_identity()        # Gets an identity matrix
        self.projection = mt.create_identity()      # Sets the projection perspective
        self.view = mt.create_identity()            # sets the camera location

        self.ZNEAR = 0.1
        self.ZFAR = 10.0
        
        # UNIFORM VALUES FOR SHADERS
        self.lightPos = [0, 5.0, 0] # -1, 5.0, -1
        self.lightColor = [1.0, 1.0, 0] # 1.0, 0.87, 0.13
        self.camera_pos = [0, 0, 0]
        self.tree_color = [1.0, 1.0, 1.0, 1.0]
        self.triangle_color = [1.0, 0.0, 1.0, 0.0]

        self.WHOLE_TREE_DEPTH = -5.0
        self.TREE_SECTION_DEPTH = -1.5
        self.TREE_DY = -2
        self.TREE_SECTION_DX = -0.25 # -0.25
        self.TREE_SECTION_ASPECT = 1.5
        
        # dimensions of the screen
        self.width = -1
        self.height = -1

        # self.loadNewMeshFiles(meshDictionary=meshDictionary)
        # Get the mesh and vertices for the mesh loaded in from json file
        # self.mesh = Mesh(self.fname)
        self.mesh = meshDictionary["Tree"]
        self.vertices = np.array(self.mesh.vertices, dtype=np.float32) # contains texture coordinates, vertex normals, vertices      
        self.texture = None
        self.vao = None
        self.vbo = None 

        # Manipulation files
        # self.meshes = [None] * 10 # fill in for meshes of the manipulations branches
        self.meshes = meshDictionary["Branches"]["Meshes"]
        self.meshDescriptions = meshDictionary["Branches"]["Description"]
        self.meshScales = meshDictionary["Branches"]["Scale"]
        self.meshRotations = meshDictionary["Branches"]["Rotation"]
        self.meshTranslations = meshDictionary["Branches"]["Translation"]
        self.meshAnswer =  meshDictionary["Branches"]["Answer"]
        self.branchProgram = None
        self.VAOs = [None] * len(self.meshes) # list of all VBOs
        self.VBOs = [None] * len(self.meshes) # list of associated VBOs
        self.branchTexture = None
        self.currentFeature = None
        self.correctFeature = False


        # GETTING THE BACKGROUND SKY BOX
        # self.skyMesh = Mesh("../obj_files/skyBox.obj") # Always the same regardless of tree File
        self.skyMesh = meshDictionary["SkyBox"]
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
        
        
        # FOR DRAWING THE BOUNDING BOX IN THE WHOLE CAMERA VIEW
        self.boundBoxProgram = None
        self.boundBoxVAO = None
        self.boundBoxVBO = None
        self.boundBoxVertices = Shapes(shape="square").vertices


        # DRAWING VALUES
        self.drawVAO = None
        self.drawVBO = None
        self.drawProgram = None
        self.drawLines = False # dtermine if to show the line
        self.drawVertices = np.zeros(3000, dtype=np.float32) # give me a set of values to declare for vbo
        self.drawCount = 0
        
        # TEXT DISPLAYING
        self.characters = {}
        self.textVAO = None
        self.textVBO = None
        self.textProgram = None
        self.displayLabels = False
        self.textVertices = np.array([[0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 1.0],
                                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        
        self.labelLines = np.zeros( 4 * 2 * 3 ) # 4 labels * 2 pts per label * 3 dimensions
        self.labelVAO = None
        self.labelVBO = None
        self.labelProgram = None
        
        # Get more information from the json file
        self.setFeatureLocations()


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
    

    def setManipulationIndex(self, index):
        if index != self.index:
            self.index = index
            self.manipulationIndex.emit(index)
            self.update()

    def setToBinAndToManipulate(self, toManipulate=False, toBin=False):
        self.toManipulate = toManipulate
        self.toBin = toBin

    def loadNewJSONFile(self, jsonData):
        self.jsonData = jsonData
        self.setFeatureLocations()

    
    def setFeatureLocations(self):
        self.budLocation = np.array(self.jsonData["Features"]["Bud"], dtype=np.float32)
        self.trunkLocation = np.array(self.jsonData["Features"]["Trunk"], dtype=np.float32)
        self.secondaryLocation = np.array(self.jsonData["Features"]["Secondary Branch"], dtype=np.float32)
        self.tertiaryLocation = np.array(self.jsonData["Features"]["Tertiary Branch"], dtype=np.float32)


    def loadNewMeshFiles(self, meshDictionary):
        if meshDictionary["Tree"] is not None:
            self.mesh = meshDictionary["Tree"]
            self.vertices = np.array(self.mesh.vertices, dtype=np.float32) # contains texture coordinates, vertex normals, vertices      
            self.texture = None
            self.vao = None
            self.vbo = None
            self.initializeTreeMesh()
        
        if meshDictionary["Branches"] is not None:
            # TO DO, Make the manipulation variable True
            self.meshes = meshDictionary["Branches"]["Meshes"]
            self.meshDescriptions = meshDictionary["Branches"]["Descriptions"]
            self.meshAnswer = meshDictionary["Branches"]["Answer"]
            self.branchProgram = None
            self.VAOs = [None] * 10 # list of all VBOs
            self.VBOs = [None] * 10 # list of associated VBOs
            self.branchTexture = None
            self.toManipulate = True
            self.initializeManipulationFiles()

        if meshDictionary["SkyBox"] is not None:
            self.skyMesh = meshDictionary["SkyBox"]
            self.skyVertices = np.array(self.skyMesh.vertices, dtype=np.float32)
            self.skyProgram = None
            self.skyTexture = None
            self.skyVAO = None
            self.skyVBO = None
            self.initializeSkyBox()

        self.update() # UPDATE THE SCREEN TO DRAW THE NEW FILES

    
    def getMeshRotation(self, rotations, cameraRotation):
        # convert to radians
        xRad = self.angle_to_radians(rotations[0])
        yRad = self.angle_to_radians(rotations[1])
        zRad = self.angle_to_radians(rotations[2])

        return mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) @ cameraRotation
        
    def getMeshTranslation(self, translation, treeTranslation):
        return np.transpose(mt.create_from_translation(translation)) @ treeTranslation
        


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
    

    def initializeManipulationFiles(self):
        gl.glUseProgram(0)
        
        if len(self.meshes) > 0: # IF NONE, THEN NO NEED TO LOAD
            # create the shader
            vertexShader = Shader("vertex", "shader.vert").shader # get out the shader value    
            fragmentShader = Shader("fragment", "shader.frag").shader

            self.branchProgram = gl.glCreateProgram()
            gl.glAttachShader(self.branchProgram, vertexShader)
            gl.glAttachShader(self.branchProgram, fragmentShader)
            gl.glLinkProgram(self.branchProgram)

            # Need to loop through all the files
            # files = self.manipulationJSON[self.manipulation]["Files"]
            for i, mesh in enumerate(self.meshes): # len(files)

                # create and bind VAOs and VBOs
                self.VAOs[i] = gl.glGenVertexArrays(1)
                gl.glBindVertexArray(self.VAOs[i])

                self.VBOs[i] = gl.glGenBuffers(1)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VBOs[i])
                
                # Load in the Mesh file and vertex
                # fname = "../obj_files/" + self.manipulation + "/" + str(files[i]["Name"])
                # self.meshes[i] = Mesh(fname)
                vertices = np.array(mesh.vertices, dtype=np.float32) # self.meshes[i]

                gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW) # gl.GL_STATIC_DRAW
                # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex.nbytes, self.vertex, gl.GL_STATIC_DRAW)

                # SET THE ATTRIBUTE POINTERS SO IT KNOWS LCOATIONS FOR THE SHADER
                stride = vertices.itemsize * 8 # times 8 given there are 8 vertices across texture coords, normals, and vertex positions before next set
                # TEXTURE COORDINATES
                gl.glEnableVertexAttribArray(0)
                gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)) 

                # NORMAL VECTORS (normals)
                gl.glEnableVertexAttribArray(1)  # SETTING THE NORMAL
                # gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(0)) # for location = 0
                gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8)) # starts at position 2 with size 4 for float32 --> 2*4

                # VERTEX POSITIONS (vertexPos)
                gl.glEnableVertexAttribArray(2) # SETTING THE VERTEX POSITIONS
                gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(20)) # starts at position 5 with size 4 --> 5*4

                # Reset the buffers 
                gl.glBindVertexArray(0)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            self.branchTexture = gl.glGenTextures(1)
            # print(f"texture ID {self.texture}")
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.branchTexture) 
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT) # will repeat the image if any texture coordinates are outside [0, 1] range for x, y values
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST) # describing how to perform texture filtering (could use a MipMap)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

            # LOAD IN THE TEXTURE IMAGE AND READ IT IN TO OPENGL
            # texImg = Image.open("../textures/bark.jpg")
            texImg = Image.open("../textures/testTexture.png")
            texData = texImg.convert("RGB").tobytes()
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 
                            0,                      # mipmap setting (can change for manual setting)
                            gl.GL_RGB,              # texture file storage format
                            texImg.width,           # texture image width
                            texImg.height,          # texture image height
                            0,                      # 0 for legacy 
                            gl.GL_RGB,              # format of the texture image
                            gl.GL_UNSIGNED_BYTE,    # datatype of the texture image
                            texData)                # data texture 
            
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindVertexArray(0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glUseProgram(0)



    def drawManipulationBranch(self):
        # Index is the index of what file should be loaded
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        if len(self.meshes) > 0: # check I have values just in case
            # Bind the texture and link the program
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.branchTexture)
            gl.glUseProgram(self.branchProgram) 

            # BIND VALUES THAT DON'T CHANGE WITH THE MANIPULATION
            lightPosLoc = gl.glGetUniformLocation(self.program, "lightPos")
            gl.glUniform3fv(lightPosLoc, 1, self.lightPos)
            lightColorLoc = gl.glGetUniformLocation(self.program, "lightColor")
            gl.glUniform3fv(lightColorLoc, 1, self.lightColor)
            projLoc = gl.glGetUniformLocation(self.program, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)
            viewLoc = gl.glGetUniformLocation(self.program, "view")
            gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            # rotation
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

            if self.wholeView: # looking at the whole tree view
                treeTranslation = np.transpose(mt.create_from_translation([0, 0, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, 0, self.TREE_SECTION_DEPTH])) 

            if self.index < len(self.meshes):
                # MODIFY BRANCH OBJECTS BASED ON FILES READ IN 
                
                rotation = self.getMeshRotation(self.meshRotations[self.index], cameraRotation) 
                translation = self.getMeshTranslation(self.meshTranslations[self.index], treeTranslation)
                scale = mt.create_from_scale(self.meshScales[self.index]) # get the scale at that index [0.1, 0.1, 0.1]
                
                model = mt.create_identity()
                model = translation @ rotation @ scale
                
                # SET SHADER PROGRAM 
                modelLoc = gl.glGetUniformLocation(self.program, "model")
                gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation
           

                gl.glBindVertexArray(self.VAOs[self.index])
                vertices = np.array(self.meshes[self.index].vertices, dtype=np.float32)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))
                self.currentFeature = self.meshDescriptions[self.index]
                # print(f"Current feature is {self.currentFeature}")

                if self.currentFeature == self.meshAnswer:
                    self.correctFeature = True
                else:
                    self.correctFeature = False
                    

                gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0)   


    def drawBinBranches(self):
                # Index is the index of what file should be loaded
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        if len(self.meshes) > 0: # check I have values just in case
            # Bind the texture and link the program
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.branchTexture)
            gl.glUseProgram(self.branchProgram) 

            # BIND VALUES THAT DON'T CHANGE WITH THE MANIPULATION
            lightPosLoc = gl.glGetUniformLocation(self.program, "lightPos")
            gl.glUniform3fv(lightPosLoc, 1, self.lightPos)
            lightColorLoc = gl.glGetUniformLocation(self.program, "lightColor")
            gl.glUniform3fv(lightColorLoc, 1, self.lightColor)
            projLoc = gl.glGetUniformLocation(self.program, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)
            viewLoc = gl.glGetUniformLocation(self.program, "view")
            gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            # rotation
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

            if self.wholeView: # looking at the whole tree view
                treeTranslation = np.transpose(mt.create_from_translation([0, 0, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, 0, self.TREE_SECTION_DEPTH])) 

            for i in range(len(self.meshes)):
                
                rotation = self.getMeshRotation(self.meshRotations[i], cameraRotation) 
                translation = self.getMeshTranslation(self.meshTranslations[i], treeTranslation)
                scale = mt.create_from_scale(self.meshScales[i]) # get the scale at that index [0.1, 0.1, 0.1]
                
                model = mt.create_identity()
                model = translation @ rotation @ scale
                
                # SET SHADER PROGRAM 
                modelLoc = gl.glGetUniformLocation(self.program, "model")
                gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation
           
                gl.glBindVertexArray(self.VAOs[i])
                vertices = np.array(self.meshes[i].vertices, dtype=np.float32)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))
                self.currentFeature = self.meshDescriptions[i]
                # print(f"Current feature is {self.currentFeature}")

                if self.currentFeature == self.meshAnswer:
                    self.correctFeature = True
                else:
                    self.correctFeature = False
                    

                gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0) 


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
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8)) # starts at position 2 with size 4 for float32 --> 2*4

        # VERTEX POSITIONS (vertexPos)
        gl.glEnableVertexAttribArray(2) # SETTING THE VERTEX POSITIONS
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(20)) # starts at position 5 with size 4 --> 5*4


    #########################################################
    # DESCRIPTION: Initializing textures for loading text on the screen
    # Code inspired by: https://learnopengl.com/In-Practice/Text-Rendering
    ########################################################
    def initializeText(self):
        # Create a text program
        gl.glUseProgram(0)
        vertexShader = Shader("vertex", "text_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "text_shader.frag").shader

        self.textProgram = gl.glCreateProgram()
        gl.glAttachShader(self.textProgram, vertexShader)
        gl.glAttachShader(self.textProgram, fragmentShader)
        gl.glLinkProgram(self.textProgram)
        gl.glUseProgram(self.textProgram)
    
        # ft = FT_Library()
        # FT_Init_FreeType(byref(ft))

        face = Face(r'C:/Windows/Fonts/arial.ttf', 0)
        # face = FT_Face()
        # FT_New_Face(ft, r'C:/Windows/Fonts/arial.ttf', 0, byref(face))
        # FT_Set_Pixel_Sizes(face, 0, 48)
        face.set_pixel_sizes(0, 48) # set the size of the font to 48 with dynamically growing widths based on height
        # if FT_Load_Char(face, 'X', freetype.FT_LOAD_RENDER):
        #     print("Failed to load glyph")
        #     return
        if face.load_char("X"):
            print("Failed to load glyph")
            return
        
        slot = face.glyph
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1) 
        # START LOOP TO INITIALIZE GLYPH TEXTURES
        for c in range(128): # want to loop through the first 128 characters
            # if FT_Load_Char(face, chr(c), freetype.FT_LOAD_RENDER):
            #     print(f"Failed to load glyph {chr(c)}")
            #     continue
            if face.load_char(chr(c)): 
                print(f"Failed to load glyph {chr(c)}")
                continue
        
            # Create a texture for the glpyh
            texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RED,                  # internal format
                slot.bitmap.width,
                slot.bitmap.rows,
                0,
                gl.GL_RED,                  # format options
                gl.GL_UNSIGNED_BYTE,
                slot.bitmap.buffer
            )

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            
            char = Character(texture, (slot.bitmap.width, slot.bitmap.rows), (slot.bitmap_left, slot.bitmap_top), slot.advance.x)
            self.characters[chr(c)] = char # store the character in a dictionary for easy access
    
        
        # create and bind the vertex attribute array for the text rendering later
        self.textVAO = gl.glGenVertexArrays(1)
        self.textVBO = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.textVAO)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.textVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.textVertices.itemsize * 6 * 4, self.textVertices, gl.GL_DYNAMIC_DRAW) 
        gl.glEnableVertexAttribArray(0) # set location = 0 in text vertex shader
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 4 * self.textVertices.itemsize, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # Bind vao and vbo for labels
        gl.glUseProgram(0) 
        # Create program, VAO and VBO for drawing the lines directly from the label to the object on the tree
        vertexShader = Shader("vertex", "simple_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "simple_shader.frag").shader
        self.labelProgram = gl.glCreateProgram()
        gl.glAttachShader(self.labelProgram, vertexShader)
        gl.glAttachShader(self.labelProgram, fragmentShader)
        gl.glLinkProgram(self.labelProgram)
        gl.glUseProgram(self.labelProgram)
        self.labelVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.labelVAO)
        self.labelVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.labelVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.labelLines.itemsize * 24, self.labelLines, gl.GL_DYNAMIC_DRAW) # 6 vertices at a time (2 end points)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * self.labelLines.itemsize, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)



    def renderText(self, text, x, y, scale):
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()
        gl.glUseProgram(self.textProgram) # activate the program
        # allows us to map the text on screen using screen coordinates
        textProject = np.transpose(mt.create_orthogonal_projection_matrix(0.0, self.width, 0.0, self.height, self.ZNEAR, self.ZFAR))
        textProjLoc = gl.glGetUniformLocation(self.textProgram, "projection")
        gl.glUniformMatrix4fv(textProjLoc, 1, gl.GL_TRUE, textProject)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self.textVAO)
        for char in text:
            # intChar = ord(char) # convert the character to a number
            character = self.characters[char]

            xpos = x + character.bearing[0] * scale # get the x value from the bearing
            ypos = y - (character.size[1] - character.bearing[1]) * scale # get the y value from bearing and scale

            w = character.size[0] * scale
            h = character.size[1] * scale

            textVertices = np.array([[xpos,     ypos + h, 0.0, 0.0],
                                    [xpos,     ypos,     0.0, 1.0],
                                    [xpos + w, ypos,     1.0, 1.0],
                                    
                                    [xpos,     ypos + h, 0.0, 0.0],
                                    [xpos + w, ypos,     1.0, 1.0],
                                    [xpos + w, ypos + h, 1.0, 0.0]], dtype=np.float32)

            # Bind the character's texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, character.texID)
            # Update the buffer
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.textVBO)
            # print("Text VBO", self.textVBO)
            # gl.glNamedBufferSubData(self.textVBO, 0, textVertices.nbytes, textVertices)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, textVertices.nbytes, textVertices)
            # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6) # number of rows for text
            x += (character.advance >> 6) * scale
            # print(f"{char} x value {x}")
        
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glPopMatrix()
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_CULL_FACE)
         



    def drawLabels(self, screenPose):
        # Loop through each label
        # Need to convert x, y of screen positions to local space coordinates
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()


        for i, label in enumerate(self.jsonData["Features"]):
            
            x, y = screenPose[i]
            start = 6 * i
            end = 6 * (i+1)
            self.renderText(text=label, x=x, y=y, scale=1.0)
            
            # find the position on the screen in local coordinates
            u,v = self.convertXYtoUV(x, y) 
            self.labelLines[start:start+3] = self.convertUVDtoXYZ(u, v, 0)[:3] # want at the 0 position on the screen 
            self.labelLines[end-3:end] = np.array(self.jsonData["Features"][label], dtype=np.float32) # locations stored in local space
        

        gl.glUseProgram(0)
        gl.glUseProgram(self.labelProgram)

        modelLoc = gl.glGetUniformLocation(self.labelProgram, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, self.model) # self.rotation

        projLoc = gl.glGetUniformLocation(self.labelProgram, "projection")
        gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection) # use the same projection values

        viewLoc = gl.glGetUniformLocation(self.labelProgram, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) # use the same location view values

        # BIND VAO AND TEXTURE
        gl.glBindVertexArray(self.labelVAO)
        # gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.skyTexture)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.labelVBO)
        gl.glNamedBufferSubData(self.labelVBO, 0, self.labelLines.nbytes, self.labelLines)
        # gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, lines.nbytes, lines)

        gl.glDrawArrays(gl.GL_LINES, 0, int(self.labelLines.size)) 
        # gl.glDrawElements(gl.GL_QUADS, int(self.skyVertices.size), gl.GL_UNSIGNED_INT, 0)

        gl.glUseProgram(0)
        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()




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


    def initializeBoundingBox(self):
        gl.glUseProgram(0)

        vertexShader = Shader("vertex", "simple_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "simple_shader.frag").shader

        self.boundBoxProgram = gl.glCreateProgram()
        gl.glAttachShader(self.boundBoxProgram, vertexShader)
        gl.glAttachShader(self.boundBoxProgram, fragmentShader)
        gl.glLinkProgram(self.boundBoxProgram)
        gl.glUseProgram(self.boundBoxProgram)

        self.boundBoxVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.boundBoxVAO)

        self.boundBoxVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.boundBoxVBO)

        # Use the skyvertices vertex counts to draw the bounding box
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.boundBoxVertices.nbytes, self.boundBoxVertices, gl.GL_STATIC_DRAW)

        stride = self.drawVertices.itemsize * 3
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)) # coordinates start at bit 20 (5*4)


    def drawBoundingBox(self):
        gl.glLoadIdentity()
        gl.glPushMatrix()
        gl.glUseProgram(0)
        
        # aspectWidth = self.width / self.height
        scaleRatio = (self.WHOLE_TREE_DEPTH - self.TREE_SECTION_DEPTH) / self.WHOLE_TREE_DEPTH
        scale = mt.create_from_scale([self.TREE_SECTION_ASPECT * scaleRatio, scaleRatio, 1]) # aspect, 1, 1
        
        moveX = -1 * (self.WHOLE_TREE_DEPTH * (self.TREE_SECTION_DX / self.TREE_SECTION_DEPTH))  # Ratio to shift should be the same as x/z for tree section
        translation = np.transpose(mt.create_from_translation([moveX, 0, self.WHOLE_TREE_DEPTH])) # -1*self.TREE_SECTION_DX, -1*self.TREE_DY
        model = translation @ scale # for rotating the modelView only want to translate and scale but not rotate

        gl.glUseProgram(self.boundBoxProgram)

        modelLoc = gl.glGetUniformLocation(self.boundBoxProgram, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation
        # print(f"Model:\n{model}")

        viewLoc = gl.glGetUniformLocation(self.boundBoxProgram, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) # use the same location view values

        projLoc = gl.glGetUniformLocation(self.boundBoxProgram, "projection")
        gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection) # use the same projection values
        # print(f"Projection:\n{self.projection}")


        # BIND VAO AND TEXTURE
        gl.glBindVertexArray(self.boundBoxVAO)
        # TO FIX
        gl.glLineWidth(2.0)
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, int(self.boundBoxVertices.size / 3))  # draw the vertices of the skybox 
        # gl.glDrawElements(gl.GL_QUADS, int(self.skyVertices.size), gl.GL_UNSIGNED_INT, 0)

        gl.glUseProgram(0)
        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()




    def initializeDrawing(self):
        gl.glUseProgram(0)

        vertexShader = Shader("vertex", "simple_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "simple_shader.frag").shader

        self.drawProgram = gl.glCreateProgram()
        gl.glAttachShader(self.drawProgram, vertexShader)
        gl.glAttachShader(self.drawProgram, fragmentShader)
        gl.glLinkProgram(self.drawProgram)

        gl.glUseProgram(self.drawProgram)

        self.drawVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.drawVAO)

        self.drawVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.drawVBO)

        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.drawVertices.nbytes, self.drawVertices, gl.GL_DYNAMIC_DRAW) # GL_STATIC_DRAW

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
        gl.glLineWidth(5.0)
        gl.glDrawArrays(gl.GL_QUADS, 0, int(self.drawVertices.size / 3))
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.drawCount) 

        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0)


    def initializeTreeMesh(self):
        # TREE SHADER
        # gl.glUseProgram(0)
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



    def paintTree(self):
        # set the perspective
        # PUT THE OBJECT IN THE CORRECT POSITION ON THE SCREEN WITH 0, 0, 0 BEING THE CENTER OF THE TREE
        # Specifically putting the object at x = 0, 0, -5.883 --> u = 0, v = 0, d = 0.5
        # Based on calculations from the projection matrix with fovy = 45, aspect = (646/616), near = 0.1, far = 10.0
        # rotate and translate the model to the correct position
        # Deal with the rotation of the object
        gl.glUseProgram(0)

        hAngle = self.angle_to_radians(self.turntable)
        vAngle = self.angle_to_radians(self.vertical)

        rotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

        if self.wholeView: # looking at the whole tree view
            translation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH])) # 0, -0, -5.883
        else:
            translation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH]))
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

        gl.glUseProgram(0)

       

    def initializeGL(self):
        # print(self.getGlInfo())
        # gl.glClearColor(0, 0.224, 0.435, 1)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        # gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        # gl.glClearColor(0.56, 0.835, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND) # for text and skybox
        gl.glDisable(gl.GL_CULL_FACE)
        # gl.glEnable(gl.GL_CULL_FACE)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
         
        self.initializeTreeMesh()

        self.initializeManipulationFiles() # Also for the binning values

        self.initializeDrawing() # initialize the places for the drawing of values

        self.initializeText()
        # self.initializeLabelBoxes()
        if self.wholeView:
            self.initializeBoundingBox()

        self.initializeSkyBox() # initialize all the skybox data

        
    def resizeGL(self, width, height):
        self.width = width 
        self.height = height # int(width // self.TREE_SECTION_ASPECT) # have a set aspect ratio

        # if not self.wholeView:
        #     print(f"Dimensions: {self.width} x {self.height}")

        side = min(width, height)
        if side < 0:
            return
        
        # gl.glViewport(0, 0, width, height)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # aspect = width / float(height)
        aspect = 0 
        if not self.wholeView:
            aspect = self.TREE_SECTION_ASPECT
        else:
            aspect = self.width / self.height

        # Set the perspective of the scene
        GLU.gluPerspective(45.0,            # FOV in y direction
                           aspect,          # FOV in x direction
                           self.ZNEAR,      # z near
                           self.ZFAR)       # z far
        self.projection = np.array(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX))
        self.projection = np.transpose(self.projection)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        # self.update()

    
    def angle_to_radians(self, angle):
        return angle * (np.pi / 180.0)



    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glLoadIdentity()
        gl.glPushMatrix()

        self.paintTree()

        if self.wholeView:
            self.drawBoundingBox() 

        if self.toManipulate and not self.wholeView:
            self.drawManipulationBranch()

        if self.toBin and not self.wholeView:
            self.drawBinBranches()

        # WANT TO DRAW THE POINTS BASED ON WHAT SOMEONE WAS DRAWING
        if self.drawLines and not self.wholeView and self.screenType == "normal":
            self.drawPruningLines()

        if not self.wholeView and self.displayLabels:
            screenPose = [(1000, 1500), # trunk
                          (2000, 900), # secondary branch
                          (250, 500), # Tertiary
                          (250, 250)] # bud
            self.drawLabels(screenPose)
            
        # DRAW THE BACKGROUND SKYBOX
        gl.glUseProgram(0)
        self.drawSkyBox()
        gl.glUseProgram(0) 

    

    # MOUSE ACTION EVENTS WITH GLWIDGET SCREEN
    def mousePressEvent(self, event) -> None:
        self.startPose = QPoint(event.pos()) # returns the last position of the mouse when clicked
    

    def mouseReleaseEvent(self, event) -> None:
        self.lastPose = QPoint(event.pos())
        if abs(self.lastPose.x() - self.startPose.x()) > 5 and abs(self.lastPose.y() - self.startPose.y()) > 5:
            self.rayDraw()
        else:
            print("Drawing Circle")
        # _ = self.rayDirection(self.lastPose.x(), self.lastPose.y())
    

    def convertXYtoUV(self, x=0, y=0):
        u = ((2 * x) / self.width) - 1.0 
        v = 1.0 - ((2 * y) / self.height)
        # print(u, v)
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
        # print(f"Local space: {local_space}")
        local_space /= local_space[3] # convert back to local space by normalizing x,y,z by w
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
        # ray /= ray[3] # divide by the w component 
        
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


    def addDrawVertices(self, drawPts):
    
        for i in range(len(drawPts)):
            # start adding at point 3*count in draw array
            start = (self.drawCount+i) * 3
            # localPt = self.convertWorldToLocal(quad[i])
            self.drawVertices[start:start+3] = drawPts[i]
            # self.draw["vertices"].extend(quad[i])
        self.drawCount += len(drawPts) # add 4 since added 4 vertices to create a quad
        # self.draw["count"] += 4


    def get_drawn_coords(self, u, v, z):
        # convert z to d
        depth = self.convertWorldtoUVD(pt=[0, 0, z])
        # if depth[2] > 1:
        #     localPt = self.convertUVDtoXYZ(u=u, v=v, d=1)
        # # print(f"UVD pt: {u}, {v}, {depth[2]}")
        # else:
        localPt = self.convertUVDtoXYZ(u=u, v=v, d=depth[2])            
        # need to divide by w value to get x, y, z
        localPt /= localPt[-1]
        return localPt[:3]
    

    def determine_draw_plane(self, startPose, endPose, u1, v1, minZ, u2, v2, maxZ):
        u3 = 0
        v3 = 0
        u4 = 0
        v4 = 0
        deltaY = endPose.y() - startPose.y()
        deltaX = endPose.x() - startPose.x()
        # Assuming the line is horizontal
        if abs(deltaY) < 20:
            u3 = u1
            v3 = v1 - 0.005
            u4 = u2
            v4 = v2 - 0.005
        # vertical line  
        elif abs(deltaX) < 20:
            u3 = u1 - 0.005
            v3 = v1
            u4 = u2 - 0.005
            v4 = v2

        # line slanted down
        elif deltaY / deltaX < 0:
            angle = math.atan2(deltaY, deltaX) # in radians
            u3 = u1 - (0.005)*math.sin(angle)
            v3 = v1 + (0.005)*math.cos(angle)
            u4 = u2 - (0.005)*math.sin(angle)
            v4 = v2 + (0.005)*math.cos(angle)
        
        else:
            angle = math.atan2(deltaY, deltaX) # in radians
            u3 = u1 - (0.005)*math.sin(angle)
            v3 = v1 - (0.005)*math.cos(angle)
            u4 = u2 - (0.005)*math.sin(angle)
            v4 = v2 - (0.005)*math.cos(angle)


        # remember that x increases as you go right and y increases as you go down
        # convert coordinates from u,v and world to local coordinates abd arrange in a cube
        #  2________4
        #  /|      /|
        # /_|_____/ |
        # 1       3 |
        # | |____|__|
        # | 6    |  8
        # | /    | /
        # |/_____|/
        # 5       7

        drawPt1 = self.get_drawn_coords(u1, v1, minZ)
        drawPt2 = self.get_drawn_coords(u1, v1, maxZ)
        drawPt3 = self.get_drawn_coords(u2, v2, maxZ)
        drawPt4 = self.get_drawn_coords(u2, v2, minZ)
        drawPt5 = self.get_drawn_coords(u3, v3, minZ)
        drawPt6 = self.get_drawn_coords(u3, v3, maxZ)
        drawPt7 = self.get_drawn_coords(u4, v4, minZ)
        drawPt8 = self.get_drawn_coords(u4, v4, maxZ)

        cubeVertices = [drawPt1, drawPt2, drawPt3, drawPt4,
                        drawPt1, drawPt2, drawPt6, drawPt5, 
                        drawPt5, drawPt6, drawPt7, drawPt8,
                        drawPt3, drawPt4, drawPt7, drawPt8,
                        drawPt2, drawPt4, drawPt6, drawPt8,
                        drawPt1, drawPt3, drawPt5, drawPt7]

        return cubeVertices   




    def rayDraw(self):
        # Use the first and last pose to get the midpoint
        
        # checking the midpoint value
        # midPt = [(self.startPose.x() + self.lastPose.x())/2, (self.startPose.y() + self.lastPose.y())/2]
        # dir = self.rayDirection(x=midPt[0], y=midPt[1])[:3]
        # print(dir)

        u1, v1 = self.convertXYtoUV(x=self.startPose.x(), y=self.startPose.y())
        u2, v2 = self.convertXYtoUV(x=self.lastPose.x(), y=self.lastPose.y())

        # print(f"Start ({self.startPose.x()}, {self.startPose.y()}), or ({u1}, {v1})")
        # print(f"End ({self.lastPose.x()}, {self.lastPose.y()}) or ({u2}, {v2})")
        # returns intersect faces in local coordinates
        intersectFaces = self.mesh.intersect_faces(u1=u1, v1=v1, u2=u2, v2=v2, projection=self.projection, view=self.view, model=self.model)
        # intersectFaces = None
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
                maxZ = np.max(depth) + 0.05 # need to offset to get in the right spot
                # print(f"Local Zs: {minZ} & {maxZ}")


                # Check if the distance is too great between values as it shouldn't be larger than 0.15 at most
                # print(maxZ - minZ)
                if maxZ - minZ > 0.1:
                    center = (maxZ + minZ) / 2
                    # print(f"Center {center}")
                    minZ = center - ((maxZ + minZ) / 5)
                    # print(f"MinZ {minZ} and MaxZ {maxZ}")
                    maxZ = center + ((maxZ + minZ) / 5)
                    

                # See the distance:
                drawPts = self.determine_draw_plane(self.startPose, self.lastPose, u1, v1, minZ, u2, v2, maxZ)

                # Need to calculate difference in u1 v1 and u2 v2 for adding a description 
                # self.addDrawVertices(drawPt1, drawPt2, drawPt3, drawPt4)
                self.addDrawVertices(drawPts)

                # UPDATE VBO TO INCORPORATE THE NEW VERTICES
                gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)

        # print(f"Total time for draw: {time.time() - start}\n")
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

                # localPt = self.convertWorldToLocal(pt)
                # print(f"Pt {pt} has local depth: {localPt[2]}")
                depth.append(pt[2]) # append the local depth to the depth
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
       
        denominator = np.dot(rayDirection, normal)
        
        if denominator <= 1e-8: # close to 0 emaning it is almost parallel
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
        delta = 1e-3 # arrived after testing as most values were negative -0.004 which is close enough to zero
        
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
            print(self.drawCount)
            start = (self.drawCount - 24) * 3
            end = self.drawCount * 3

            # print(f"Delete vertices from indices {start}:{end}")
            # need to replace all the values from drawCount
            self.drawVertices[start:end] = np.zeros(end - start)
            self.drawCount -= 24  # becuase of the cube vertices count
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

    def addLabels(self, checked=False):
        self.displayLabels = checked
        self.update()

    def setManipulationIndex(self, index=0):
        self.index = index
        self.update()
    

# QWidget 
# QtOpenGL.QMainWindow 
class Window(QMainWindow):

    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        # Practice loading in data
        self.fname = "textureTree.obj"
        self.jsonData = JSONFile("objFileDescription.json", "o").data

        # QtOpenGL.QMainWindow.__init__(self)
        # self.resize(1000, 1000)
        self.setGeometry(100, 100, 1000, 1000)
        self.setWindowTitle("TEST Window")

        self.index = 0
        self.correctFeature = False
             
        
        self.screen_width = 3    # How many rows to cover
        # getting the main screen
        self.manipulation = {
            "Display": False,
            "JSON": self.jsonData["Manipulation Files"],
            "Directory": "manipulation"
        }

        self.binValues = ["SELECT", "Don't Prune", "Prune Back", "Prune Completely"] #
        self.binAnswers = ["SELECT", "SELECT", "SELECT"]
        self.binIndices = [0, 0, 0]

        self.screenType = "bin" # Options: bin, submit, manipulation, normal 
        
        self.previousScreen = self.screenType
        self.toManipulate = True
        self.submitScreen = False
        self.nextScreenManipualte = False
        self.skyBoxMesh = None # skybox is the same for every window

        self.manipulationDir = "vigor_bins"
        # Initial meshes to load
        self.meshDictionary = self.load_mesh_files(treeFile="textureTree.obj", branchFiles=self.jsonData["Manipulation Files"], manipDirectory=self.manipulationDir, skyBoxFile="skyBox.obj")

        self.skyBoxMesh = self.meshDictionary["SkyBox"]

        self.loadScreen()   # LOAD THE SCREEN 
        

    def loadScreen(self):
        # RESET THE WIDGET
        print("LOADING SCREEN")
        self.central_widget = QWidget() # GLWidget()
        self.layout = QGridLayout(self.central_widget)
        self.hLayout = QHBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget) 

        # LOAD LEFT AND RIGHT SIDE OF SCREEN
        self.leftSideScreen()
        self.rightSideScreen()


    # Loading the mesh files for each value
    # INPUT:
    #   - treeFile: String of file name for the tree obj file
    #   - branchFiles: list of dictionaries containing file name and information about the branches
    #   - manipDirectory: The corresponding manipulation directory for the branchFiles
    #   - skyBoxFile: String of file name for skybox obj file
    #
    # OUTPUT:
    #   - Mesh files for tree, branches, and skybox if file is loaded in

    def load_mesh_files(self, treeFile = None, branchFiles = None, manipDirectory = None, skyBoxFile = None):
        directory = "../obj_files/"
        treeMesh = None
        skyBoxMesh = None
        branchMeshes = []
        scales = []
        rotations = []
        translations = []
        featureDescription = []
        
        branches = {
            "Meshes": branchMeshes, 
            "Description": featureDescription, 
            "Scale": scales,
            "Rotation": rotations,
            "Translation": translations,
            "Answer": ""
        }

        if treeFile is not None:
            fname = directory + treeFile
            treeMesh = Mesh(fname)
        
        if branchFiles is not None:
            for branch in branchFiles[manipDirectory]["Files"]:
                fname = directory + manipDirectory + "/"+ str(branch["File Name"]) # object name
                featureDescription.append(branch["Feature Description"])
                branchMeshes.append(Mesh(fname))

                # change the values of the scale or translation
                scales.append(branch["Scale"])
                rotations.append(branch["Rotate"])
                translations.append(branch["Translate"])
            # Populate the mesh
            branches["Meshes"] = branchMeshes
            branches["Scale"] = scales
            branches["Rotation"] = rotations
            branches["Translation"] = translations
            branches["Description"] = featureDescription
            branches["Answer"] = branchFiles[manipDirectory]["Answer"] # looking if the correct answer
        
        if skyBoxFile is not None:
            fname = directory + skyBoxFile
            skyBoxMesh = Mesh(fname)
        
        meshDictionary = {
            "Tree": treeMesh,
            "SkyBox": skyBoxMesh,
            "Branches": branches
        }

        return meshDictionary
    

    # LEFT HAND SIDE DEPENDENT ON WHAT TYPE OF SCREEN WE NEED TO SEE
    def leftSideScreen(self):
        # SET THE SCREEN SIZE BASED ON IF A MANIPULATION TASK OR NOT
        if self.screenType == "manipulation":
            self.manipulationScreen()

        elif self.screenType == "submit":
            self.submitButtonScreen()
        
        elif self.screenType == "bin":
            self.binScreen()
        else:
            self.loadTreeSectionScreen()


    ########################################################################
    # LOAD WHAT IS ON THE RIGHT HAND SIDE OF THE SCREEN (Does not change by screen)
    # This includes:
    #   - whole tree view
    #   - Your task bask
    #   - Progress Bar
    #######################################################################
    def rightSideScreen(self):
        self.viewGL = Test(wholeView=True, 
                           fname=self.fname, 
                           meshDictionary=self.meshDictionary,
                           jsonData=self.jsonData["Tree Files"][self.fname],
                           manipulation=self.manipulation)
        self.viewGL.setFixedSize(900, 700)
        self.layout.addWidget(self.viewGL, 0, 2, 1, 1) # 1, 2, 1, 1
        self.hSlider.valueChanged.connect(self.viewGL.setTurnTableRotation) # Connect the vertical and horizontal camera sliders to the view screen
        self.viewGL.turnTableRotation.connect(self.hSlider.setValue)

        self.vSlider.valueChanged.connect(self.viewGL.setVerticalRotation)
        self.viewGL.verticalRotation.connect(self.vSlider.setValue)
        

        # YOUR TASK BOX
        # Create a QFrame for the directory and buttons column
        self.textFrame = QFrame(self.central_widget) # self.central_widget
        self.textFrame.setFrameShape(QFrame.Shape.Box)
        self.textFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.textFrame, 1, 2, 1, 1)  # Row 1, Column 1, Span 1 row and 1 column

        # Create a QVBoxLayout for the directory and buttons column
        self.directory_layout = QVBoxLayout(self.textFrame)

        # Create a QLabel to display the directory
        self.directory_label = QLabel("Your Task:")
        self.directory_label.setStyleSheet("font-size: 50px;" "font:bold")

        self.directory_layout.addWidget(self.directory_label)

        # Create a QLabel to display the task description
        # QLabel should be filled in with json data
        self.task_label = QLabel("Manipulate the slider until the branch is too vigorous\n and needs to be removed entirely")
        self.task_label.setStyleSheet("font-size: 35px;")
        self.directory_layout.addWidget(self.task_label)
    
        self.progressFrame = QFrame(self.central_widget) # self.central_widget
        self.progressFrame.setFrameShape(QFrame.Shape.Box)
        self.progressFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.progressFrame, 2, 2, 2, 1)  # Row 1, Column 1, Span 1 row and 1 column

        self.progress_layout = QVBoxLayout(self.progressFrame)
        self.progress_label = QLabel("Your Progress:")
        self.progress_label.setStyleSheet("font-size: 50px;" "font:bold")

        self.progress_layout.addWidget(self.progress_label)



    def createSlider(self, camera=True, horizontal=True):
        if horizontal:
            slider = QSlider(Qt.Horizontal)
        else:
            slider = QSlider(Qt.Vertical)
        # slider.setRange(0, 360) # 0 - 360*16
        if camera:
            slider.setRange(-30, 30)
            slider.setSingleStep(1) # 
            # slider.setPageStep(10)
            slider.setPageStep(5)
            slider.setTickPosition(QSlider.TicksBelow)
        else:
            slider.setRange(0, 10)
            slider.setSingleStep(1) # 
            # slider.setPageStep(10)
            slider.setPageStep(1)
            slider.setTickPosition(QSlider.TicksBelow) 
        return slider

    ########################################################################
    # LOAD THE TREE SECTION ON THE LEFT HAND SIDE OF THE SCREEN
    # This includes:
    #   - undo button
    #   - labels button
    #   - submit button (if not in submit screen or manipulation screen)
    #   - sliders to control the camera
    #######################################################################
    def loadTreeSectionScreen(self):
      
        self.glWidgetTree = Test(wholeView=False, 
                                fname=self.fname,
                                meshDictionary=self.meshDictionary, 
                                jsonData=self.jsonData["Tree Files"][self.fname],  # NEEd to change eventually
                                manipulation=self.manipulation,
                                screenType=self.screenType)
        
        # set the index if it is empty to current index
        # only would draw for manipulation screen
        self.glWidgetTree.setManipulationIndex(self.index) 
        
        if self.screenType == "normal":
            self.screen_width = 3
            self.glWidgetTree.setToBinAndToManipulate() # default to False
        
        elif self.screenType == "bin":
            self.screen_width = 2
            self.glWidgetTree.setToBinAndToManipulate(toBin=True)
        
        elif self.screenType == "manipulation":
            self.screen_width = 2
            self.glWidgetTree.setToBinAndToManipulate(toManipulate=True)
        

        else: # submit should use previous manipulation value
            self.screen_width = 2
        
        # self.glWidget.setFixedSize(2820, 1850) # can I find the aspect
        # self.layout.addWidget(self.glWidget)
        self.layout.addWidget(self.glWidgetTree, 0, 1, self.screen_width, 1) # r=0, c=1, rs = 3, cs = 1

        # UNDO BUTTON
        self.undoButton = QPushButton("Undo")
        self.undoButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.undoButton.setFixedSize(300, 100)
        self.undoButton.clicked.connect(self.glWidgetTree.undoDraw)
        self.hLayout.addWidget(self.undoButton)
        
        # LABEL BUTTON
        self.labelButton = QPushButton("Labels On") # Make a blank button
        self.labelButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.labelButton.setCheckable(True)
        self.labelButton.setFixedSize(300, 100)
        self.labelButton.clicked.connect(self.labelButtonClicked)
        self.hLayout.addWidget(self.labelButton)

        # SUBMIT BUTTON
        if self.screenType == "normal": # should only show when both are false
            self.submitButton = QPushButton("Submit") # Make a blank button
            self.submitButton.setStyleSheet("font-size: 50px;" "font:bold")
            self.submitButton.clicked.connect(self.submitButtonClicked) # TO CONNECT THE BUTTON WITH SCREEN
            self.submitButton.setFixedSize(300, 100)
            self.hLayout.addWidget(self.submitButton)

        
        # self.layout.addWidget(self.labelButton, 0, 1, 1, 1)
        self.layout.addLayout(self.hLayout, 0, 1, Qt.AlignTop | Qt.AlignLeft)

        # VERTICAL SLIDER
        self.vSlider = self.createSlider(camera=True, horizontal=False)
        self.vSlider.valueChanged.connect(self.glWidgetTree.setVerticalRotation)
        self.glWidgetTree.verticalRotation.connect(self.vSlider.setValue)
        # self.layout.addWidget(self.vSlider)
        self.layout.addWidget(self.vSlider, 0, 0, self.screen_width, 1) # 0, 0, 3, 1
        
        # HORIZONTAL SLIDER
        self.hSlider = self.createSlider(camera=True, horizontal=True)
        self.hSlider.valueChanged.connect(self.glWidgetTree.setTurnTableRotation)
        self.glWidgetTree.turnTableRotation.connect(self.hSlider.setValue)
        self.layout.addWidget(self.hSlider, self.screen_width, 1, 1, 1) # 3 1 1 1


    
    def manipulationScreen(self):
        
        self.loadTreeSectionScreen() # LOAD THE TREE SECTION

        self.manipulationFrame = QFrame(self.central_widget) # self.central_widget
        self.manipulationFrame.setFrameShape(QFrame.Shape.Box)
        self.manipulationFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.manipulationFrame, self.screen_width+1, 1, 1, 1) # span the screen width but start below the sliders

        self.manipLayout = QVBoxLayout(self.manipulationFrame)

        # TO CHANGE THE NAME OF THE SLIDER BASED ON WHAT WE ARE MANIPULATING
        self.manipulation_label = QLabel("Manipulation Slider:") 
        self.manipulation_label.setStyleSheet("font-size: 50px;" "font:bold")
        self.manipLayout.addWidget(self.manipulation_label)

        self.manipulationSlider = self.createSlider(camera=False, horizontal=True)
        self.manipulationSlider.valueChanged.connect(self.glWidgetTree.setManipulationIndex)

        # WHICH SLIDER TO USE WILL CHANGE!
        self.glWidgetTree.manipulationIndex.connect(self.manipulationSlider.setValue)
        self.manipLayout.addWidget(self.manipulationSlider) # 3 1 1 1

        # submit button
        self.submitButton = QPushButton("Submit") # Make a blank button
        self.submitButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.submitButton.clicked.connect(self.submitButtonClicked)
        # self.submitButton.setCheckable(True)
        self.submitButton.setFixedSize(300, 100)
        self.manipLayout.addWidget(self.submitButton)

    
    
    def submitButtonScreen(self):
        
        self.submitScreen = False # RESET THE SCREEN TO FALSE SO IT LOADS A DIFFERENT SCREEN UPON RELOAD
        self.loadTreeSectionScreen() # LOAD THE TREE SECTION
        # self.glWidgetTree.index = self.index
        
        # NEED TO RELOAD THE SCREEN TO DISPLAY THE TEXT ON THE BOTTOM!
        self.submitFrame = QFrame(self.central_widget) # self.central_widget
        self.submitFrame.setFrameShape(QFrame.Shape.Box)
        self.submitFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.submitFrame, self.screen_width+1, 1, 1, 1) # span the screen width but start below the sliders


        # getting the answer
        text = ""
        self.submitLayout = QVBoxLayout(self.submitFrame)
        self.submit_label = QLabel(text)
        if self.glWidgetTree.toManipulate:
            if self.correctFeature:
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Correct"]
                self.submitButton = QPushButton("Next") 
                self.submitButton.clicked.connect(self.nextPageButtonClicked)
            else:
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Incorrect"]
                self.submitButton = QPushButton("Retry") 
                self.submitButton.clicked.connect(self.retryButtonClicked)
        
        elif self.glWidgetTree.toBin: # CHECK THE BIN ANSWERS
            correct = self.jsonData["Manipulation Files"][self.manipulationDir]["Answer"]
            if self.compareBinAnswers(correct, self.binAnswers):
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Correct"]
                self.submitButton = QPushButton("Next") 
                self.submitButton.clicked.connect(self.nextPageButtonClicked)
            else:
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Incorrect"]
                self.submitButton = QPushButton("Retry") 
                self.submitButton.clicked.connect(self.retryButtonClicked)


        self.submit_label = QLabel(text) # Want to change the text to say "Vigor
        self.submit_label.setStyleSheet("font-size: 50px;" "font:bold")
       
        self.submitButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.submitButton.setFixedSize(300, 100)
    
        self.submitLayout.addWidget(self.submit_label)
        self.submitLayout.addWidget(self.submitButton)
 
    

    def binScreen(self):
        self.loadTreeSectionScreen()

        # self.binFrame
        self.binFrame = QFrame(self.central_widget) # self.central_widget
        self.binFrame.setFrameShape(QFrame.Shape.Box)
        self.binFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.binFrame, self.screen_width+1, 1, 1, 1) # Where on the screen we add
        self.binLayout = QGridLayout(self.binFrame)
        
        # Setting the font
        font = QFont()
        font.setPointSize(font.pointSize()+2)
        # Need to add a QLayout
        # Want the branch name on top of the combo box option 
        self.binLabel = QLabel("Branch 1")
        self.binLabel.setStyleSheet("font-size: 50px;" "font:bold")
        self.binLayout.addWidget(self.binLabel, 1, 1, Qt.AlignBottom | Qt.AlignCenter) # where in the small table we add
        
        self.dropDown = QComboBox()
        self.dropDown.setFixedSize(500, 100)
        self.dropDown.setFont(font)
        self.dropDown.addItems(self.binValues)
        self.dropDown.setCurrentIndex(self.binIndices[0])
        self.dropDown.activated.connect(self.dropDownTextSelected)
        self.binLayout.addWidget(self.dropDown, 2, 1, Qt.AlignTop | Qt.AlignCenter)

        # Second dropdown box
        self.binLabel2 = QLabel("Branch 2")
        self.binLabel2.setStyleSheet("font-size: 50px;" "font:bold")
        self.binLayout.addWidget(self.binLabel2, 1, 2, Qt.AlignBottom | Qt.AlignCenter)
        
        self.dropDown2 = QComboBox()
        self.dropDown2.setFixedSize(500, 100)
        self.dropDown2.setFont(font)
        self.dropDown2.addItems(self.binValues)
        self.dropDown2.setCurrentIndex(self.binIndices[1])
        self.dropDown2.activated.connect(self.dropDownTextSelected)
        self.binLayout.addWidget(self.dropDown2, 2, 2, Qt.AlignTop | Qt.AlignCenter)

        # 3rd drop down
        self.binLabel3 = QLabel("Branch 3")
        self.binLabel3.setStyleSheet("font-size: 50px;" "font:bold")
        self.binLayout.addWidget(self.binLabel3, 1, 3, Qt.AlignBottom | Qt.AlignCenter)
        
        self.dropDown3 = QComboBox()
        self.dropDown3.setFixedSize(500, 100)
        self.dropDown3.setFont(font)
        self.dropDown3.addItems(self.binValues)
        self.dropDown3.setCurrentIndex(self.binIndices[2]) # SET THE VALUE TO PREVIOUS RESULTS
        self.dropDown3.activated.connect(self.dropDownTextSelected)
        self.binLayout.addWidget(self.dropDown3, 2, 3, Qt.AlignTop | Qt.AlignCenter)
       
        self.dropDownTextSelected("") # grab the current text in the dropdown menus

        self.submitButton = QPushButton("Submit") # Make a blank button
        self.submitButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.submitButton.clicked.connect(self.submitButtonClicked)
        self.submitButton.setFixedSize(300, 100)
        self.binLayout.addWidget(self.submitButton, 3, 1)
    
    
    def dropDownTextSelected(self, _):
        self.binAnswers[0] = self.dropDown.currentText()
        self.binIndices[0] = self.dropDown.currentIndex()

        self.binAnswers[1] = self.dropDown2.currentText()
        self.binIndices[1] = self.dropDown2.currentIndex()

        self.binAnswers[2] = self.dropDown3.currentText()
        self.binIndices[2] = self.dropDown3.currentIndex()

        print(f"Different text selected: {self.binAnswers}")
    
    def setDropDownValues(self):
        self.dropDown.setCurrentIndex(self.binIndices[0])
        self.dropDown2.setCurrentIndex(self.binIndices[1])
        self.dropDown3.setCurrentIndex(self.binIndices[2])
    

    def submitButtonClicked(self):
        print("SUBMIT BUTTON SELECTED")
        if self.screenType == "manipulation" or self.screenType == "bin":
            self.previousScreen = self.screenType
            self.correctFeature = self.glWidgetTree.correctFeature
            self.index = self.glWidgetTree.index

        self.screenType = "submit"
        self.loadScreen() 

    
    def retryButtonClicked(self):
        print(f"RETRY BUTTON SELECTED {self.previousScreen}")
        self.screenType = self.previousScreen
        self.loadScreen()

    # Need to load the next immediate screen
    def nextPageButtonClicked(self):
        print("NEXT BUTTON SELECTED")
        self.screenType = "normal" # NEED TO CHANGE BASED ON WHAT THE JSON FILE SAYS PAGE SHOULD BE
        self.glWidgetTree.toManipulate = False
        self.index = 0 # set index to 0
        self.loadScreen()

    
    def compareBinAnswers(self, answers, values):
        for answer, value in zip(answers, values):
            if answer != value:
                return False 
        return True
    

    def labelButtonClicked(self):
        checked = True
        if self.labelButton.isChecked():
            self.labelButton.setText("Labels Off")
            checked = True 
        else:
            self.labelButton.setText("Labels On")
            checked = False
        self.glWidgetTree.addLabels(checked) # activate the label check
        


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