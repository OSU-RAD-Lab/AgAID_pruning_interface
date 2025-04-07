#!/usr/bin/env python3
import sys
sys.path.append('../')

import os
os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"


from PySide6 import QtCore, QtGui, QtOpenGL

from PySide6.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QComboBox, QProgressBar, QRadioButton
    # QOpenGLWidget

from PySide6.QtOpenGLWidgets import QOpenGLWidget

from PySide6.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint, QCoreApplication, QPoint

# from PySide6.QtOpenGL import QGLWidget, QGLContext

from PySide6.QtGui import QFont
# from PySide6.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4, QSurfaceFormat, QPainter,
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
        self.toBin = False
        self.toPruneBin = False
        self.toPrune = False
        self.toBinFeatures = False
        self.termDraw = False

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
        self.lightPos = [0, 3.0, 0] # -1, 5.0, -1
        self.lightColor = [1.0, 1.0, 1.0] # 1.0, 0.87, 0.13 # 1.0, 1.0, 0
        self.camera_pos = [0, 0, 0]
        self.tree_color_dark = [0.278, 0.219, 0.227] 
        self.red_color = [1, 0, 0]
        self.tree_color_light =  [0.447, 0.360, 0.259] # [0.670, 0.549, 0.416]

        self.WHOLE_TREE_DEPTH = -5.0
        self.TREE_SECTION_DEPTH = -1.5
        self.TREE_DY = -2
        self.TREE_SECTION_DX = -0.25 # -0.25
        self.TREE_SECTION_ASPECT = 1.5
        

        self.cutNumber = 0 # keeping track of sequence of cuts

        # can use to scale
        self.MAX_WIDTH = 2820
        self.MAX_HEIGHT = 1275 # to change
        
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

        
        self.vigorColor = [255.0/255.0, 95.0/255.0, 31.0/255.0]
        self.spacingColor = [33.0/255.0, 252.0/255.0, 13.0/255.0]
        self.canopyColor = [0.0/255.0, 240.0/255.0, 255.0/255.0]
        self.penColor = self.vigorColor

        self.decision = "" # "Vigor"
        self.crossYMenu = False # don't display the 

        # for documenting the decisions users make

        self.cutSequenceDict = {
            "Rule": [],
            "Sequence": [],
            "Vertices": [],
            "Cut Time": [],
            "Rule Time": [],
            "Between Cut Time": []
        }
        
        
        self.crossyTextBounds = {
            "Branch Vigor": [], 
            "Canopy Cut": [],
            "Bud Spacing": []
        }

        # Manipulation files
        # self.meshes = [None] * 10 # fill in for meshes of the manipulations branches
        self.meshes = meshDictionary["Branches"]["Meshes"]
        self.meshDescriptions = meshDictionary["Branches"]["Description"]
        self.pruneDescription = meshDictionary["Branches"]["Prunes"]
        self.meshScales = meshDictionary["Branches"]["Scale"]
        self.meshRotations = meshDictionary["Branches"]["Rotation"]
        self.meshTranslations = meshDictionary["Branches"]["Translation"]
        self.meshAnswer =  meshDictionary["Branches"]["Answer"]
        self.branchProgram = None
        self.VAOs = [None] * len(self.meshes) # list of all VBOs
        self.pruneVAOs = [None] * len(self.meshes)
        self.VBOs = [None] * len(self.meshes) # list of associated VBOs
        self.pruneVBOs = [None] * len(self.meshes)
        self.branchTexture = None
        self.currentFeature = None
        self.correctFeature = False

        self.wantedFeature = None

        self.pruneBranches = ["Don't Prune"] * len(self.meshes) # Do I want the prune to be visible? 


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
        
        # General cube I can manipulate as a prune line
        self.pruneLines = np.array([1.000000, 1.000000, -1.000000,
                                    -1.000000, 1.000000, -1.000000, 
                                    -1.000000, 1.000000, 1.000000,
                                    1.000000, 1.000000, 1.000000,
                                    1.000000, -1.000000, 1.000000,
                                    1.000000, 1.000000, 1.000000, 
                                    -1.000000, 1.000000, 1.000000,
                                    -1.000000, -1.000000, 1.000000,
                                    -1.000000, -1.000000, 1.000000,
                                    -1.000000, 1.000000, 1.000000,
                                    -1.000000, 1.000000, -1.000000,
                                    -1.000000, -1.000000, -1.000000,
                                    -1.000000, -1.000000, -1.000000,
                                    1.000000, -1.000000, -1.000000,
                                    1.000000, -1.000000, 1.000000,
                                    -1.000000, -1.000000, 1.000000,
                                    1.000000, -1.000000, -1.000000,
                                    1.000000, 1.000000, -1.000000,
                                    1.000000, 1.000000, 1.000000,
                                    1.000000, -1.000000, 1.000000,
                                    -1.000000, -1.000000, -1.000000,
                                    -1.000000, 1.000000, -1.000000,
                                    1.000000, 1.000000, -1.000000,
                                    1.000000, -1.000000, -1.000000], dtype=np.float32) # making a cube

        self.pruneVAO = None
        self.pruneVBO = None
        self.pruneProgram = None

        self.testCube = np.array([0, 2.146, 0, # trunk
                                  0.20556462, 2.06136332-0.05, 1.28046104, # trunk
                                  -0.5038, 1.963, -0.004,
                                  0.13779826, 2.02686186-0.05, 1.28046104,
                                  0.4273, 1.922, -0.07807,                                  
                                  0.326576,   2.03179064-0.05, 1.28046104,
                                  0.1512, 1.994, -0.02245,
                                  0.28482026, 1.99236039, 1.28562765 
                                 ], dtype=np.float32)
        # [0.20556462 2.06136332 1.28046104 1.        ]
        # [0.13779826 2.02686186 1.28046104 1.        ]
        # [0.326576   2.03179064 1.28046104 1.        ]
        # [0.30721418 1.99236039 1.28046104 1.        ]

        # np.array([-0.5, 1.5, 0,
        #           0.5, 1.5, 0,
        #           0, 2, 0
        #           ], dtype=np.float32)


        # FOR DRAWING THE BOUNDING BOX IN THE WHOLE CAMERA VIEW
        self.boundBoxProgram = None
        self.boundBoxVAO = None
        self.boundBoxVBO = None
        self.boundBoxVertices = Shapes(shape="square").vertices


        self.drawTerms = [False] * len(self.meshes) # Array of what term to highlight/draw
        


        # DRAWING VALUES
        self.drawVAO = None
        self.drawVBO = None
        self.drawProgram = None
        self.drawLines = False # determine if to draw lines
        self.drawVertices = np.zeros(3600, dtype=np.float32) # give me a set of values to declare for vbo
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

    def setScreenProperties(self, screenType, toManipulate=False, toBin=False, toBinFeatures=False, toPrune=False, toPruneBin=False, drawLines=False, termDraw=False):
        self.toManipulate = toManipulate
        self.toBin = toBin
        self.toBinFeatures = toBinFeatures
        self.toPrune = toPrune
        self.toPruneBin = toPruneBin
        self.drawLines = drawLines
        self.termDraw = termDraw
        self.screenType = screenType

    def loadNewJSONFile(self, jsonData):
        self.jsonData = jsonData
        self.setFeatureLocations()

    def setFeatureLocations(self):
        self.budLocation = np.array(self.jsonData["Features"]["Bud"], dtype=np.float32)
        self.trunkLocation = np.array(self.jsonData["Features"]["Trunk"], dtype=np.float32)
        self.secondaryLocation = np.array(self.jsonData["Features"]["Secondary Branch"], dtype=np.float32)
        self.tertiaryLocation = np.array(self.jsonData["Features"]["Tertiary Branch"], dtype=np.float32)


    def loadNewMeshFiles(self, meshDictionary):
        print("LOADING NEW MESH DICTIONARY VALUES")
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
            self.meshDescriptions = meshDictionary["Branches"]["Description"]
            self.meshAnswer = meshDictionary["Branches"]["Answer"]
            self.meshScales = meshDictionary["Branches"]["Scale"]
            self.meshRotations = meshDictionary["Branches"]["Rotation"]
            self.meshTranslations = meshDictionary["Branches"]["Translation"]
            self.branchProgram = None
            self.VAOs = [None] * len(self.meshes) # list of all VBOs
            self.VBOs = [None] * len(self.meshes) # list of associated VBOs
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
        

    def calculateLabelPlacement(self, position, translation, rotation, scale):
        """
        position: x, y, z coordinate read in from json file
        translation: 4x4 translation matrix
        rotation: 4x4 rotation matrix
        scale: 4x4 scale matrix
        @return x,y screen coordinate
        """
        # need to use the conversion between uv to screen
        label = np.ones(4)
        label[:3] = position
        # need to multiply by scale and rotation
        model = translation @ rotation @ scale
        mvp = self.projection @ self.view @ model
        label_pos = mvp @ np.transpose(label)
        label_pos /= label_pos[3] # normalize the value

        x, y = self.convertUVtoScreenCoords(label_pos[0], label_pos[1])
        return x, y


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
            texImg = Image.open("../textures/bark.jpg")
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
            treeColorLoc = gl.glGetUniformLocation(self.program, "color")
            gl.glUniform3fv(treeColorLoc, 1, self.tree_color_dark) # self.tree_color_dark

            projLoc = gl.glGetUniformLocation(self.program, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)
            viewLoc = gl.glGetUniformLocation(self.program, "view")
            gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            # rotation
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

            # vertical = mt.create_from_x_rotation(vAngle) # vAngle
            # horizontal = mt.create_from_x_rotation(hAngle)

            if self.wholeView: # looking at the whole tree view
                treeTranslation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH]))
                # negTreeTranslation = np.transpose(mt.create_from_translation([0, 0, -self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH])) 
                # negTreeTranslation = np.transpose(mt.create_from_translation([-self.TREE_SECTION_DX, 0, -self.TREE_SECTION_DEPTH])) 


            if self.index < len(self.meshes):
                
                # rotation = self.getMeshRotation(self.meshRotations[self.index], cameraRotation) 
                # translation = self.getMeshTranslation(self.meshTranslations[self.index], treeTranslation)
                scale = mt.create_from_scale(self.meshScales[self.index])

                # # How much to translate the branch to get it sitting on the tree
                # if not self.wholeView:
                #     dx = self.meshTranslations[self.index][0]
                #     dy = self.meshTranslations[self.index][1] - self.TREE_DY # Need to compensate for the move up. We don't want the branch translated that far. 
                #     dz = self.meshTranslations[self.index][2]
                # else:

                # get the translations 
                dx = self.meshTranslations[self.index][0]
                dy = self.meshTranslations[self.index][1] # Need to compensate for the move up. We don't want the branch translated that far. 
                dz = self.meshTranslations[self.index][2]
                # print(f"dx: {dx}, dy: {dy}, dz: {dz}")

                # Need to resolve some translations when a rotation is involved
                xRad = self.angle_to_radians(self.meshRotations[self.index][0])
                yRad = self.angle_to_radians(self.meshRotations[self.index][1])
                zRad = self.angle_to_radians(self.meshRotations[self.index][2])
                # print(f"Angle X: {xRad}, Angle Y: {yRad}, Angle Z: {zRad}")


                branchTranslation = np.transpose(mt.create_from_translation([dx, dy - self.TREE_DY, dz]))
                # branchTranslation = np.transpose(mt.create_from_translation([dx, dy, dz]))
                branchRotation = mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) 
                
                # rotations = vertical @ horizontal @ branchRotation
                model = mt.create_identity()
                model = treeTranslation @ cameraRotation @ branchTranslation @ branchRotation @ scale
                # model = treeTranslation @ cameraRotation @ branchTranslation @ branchRotation @ scale

                # SET SHADER PROGRAM 
                modelLoc = gl.glGetUniformLocation(self.program, "model")
                gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation
           
                gl.glBindVertexArray(self.VAOs[self.index])
                vertices = np.array(self.meshes[self.index].vertices, dtype=np.float32)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))

                # Get the current branch feature and pruning description for the branch
                self.currentFeature = self.meshDescriptions[self.index]
                self.pruneFeature = self.pruneDescription[self.index] 
                
                # print(f"Current feature is {self.currentFeature}")
                if self.currentFeature == self.meshAnswer:
                    self.correctFeature = True
                else:
                    self.correctFeature = False

                # HAVE IT RENDER TEXT OF THE VIGOR LEVEL & PRUNING DECISION
                # Do NOT want the answers appearing in the manipulation check
                if self.screenType == "scale" and not self.wholeView:
                    vigorText = f"Branch Feature: {self.currentFeature}"
                    pruneText = f"Pruning Description: {self.pruneFeature}"
                    _ = self.renderText(vigorText, x=450, y=200, scale=0.35, color=[1, 0.27, 0]) # 1400, 650  [1, 0.477, 0.706]
                    _ = self.renderText(pruneText, x=450, y=170, scale=0.35, color=[1, 0.27, 0]) # 1400, 650

                # elif self.screenType = "bin" and not self.wholeView:s
                #     self.renderText("Branch", x=1200, y=820, scale=0.85, color=[1, 0, 0])

                gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0)   


    

    def drawTermSections(self):
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        highlightColor = [255/255, 105/255, 180/255] 
        if self.termDraw and len(self.meshes) > 0: # check I have values just in case
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

            treeColorLoc = gl.glGetUniformLocation(self.program, "color")
            gl.glUniform3fv(treeColorLoc, 1, highlightColor)

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            # rotation
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

            if self.wholeView: # looking at the whole tree view
                treeTranslation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH])) 

            
            for i in range(len(self.meshes)):
                
                # rotation = self.getMeshRotation(self.meshRotations[i], cameraRotation) 
                # translation = self.getMeshTranslation(self.meshTranslations[i], treeTranslation)
                dx = self.meshTranslations[i][0]
                dy = self.meshTranslations[i][1] - self.TREE_DY# Need to compensate for the move up. We don't want the branch translated that far. 
                dz = self.meshTranslations[i][2]
                
                branchTranslation = np.transpose(mt.create_from_translation([dx, dy, dz]))
                # # negBranchTranslation = np.transpose(mt.create_from_translation(-1 * np.array(self.meshTranslations[self.index])))
                xRad = self.angle_to_radians(self.meshRotations[i][0])
                yRad = self.angle_to_radians(self.meshRotations[i][1])
                zRad = self.angle_to_radians(self.meshRotations[i][2])
                branchRotation = mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) 

                scale = mt.create_from_scale(self.meshScales[i]) # get the scale at that index [0.1, 0.1, 0.1]
                
                model = mt.create_identity()
                # Need the branch to stay on the tree and then rotate with it
                # model = translation @ rotation @ scale # translation @ rotation @ scale
                model = treeTranslation @ cameraRotation @ branchTranslation @ branchRotation @ scale
                # SET SHADER PROGRAM 
                modelLoc = gl.glGetUniformLocation(self.program, "model")
                gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation
           
                gl.glBindVertexArray(self.VAOs[i])
                vertices = np.array(self.meshes[i].vertices, dtype=np.float32)

                if self.drawTerms[i]:
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))                 

                gl.glBindVertexArray(0) # unbind the vao

        gl.glPopMatrix()
        gl.glUseProgram(0)




    def drawBinBranches(self):
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
            treeColorLoc = gl.glGetUniformLocation(self.program, "color")
            gl.glUniform3fv(treeColorLoc, 1, self.tree_color_dark)

            projLoc = gl.glGetUniformLocation(self.program, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)
            viewLoc = gl.glGetUniformLocation(self.program, "view")
            gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            # rotation
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

            if self.wholeView: # looking at the whole tree view
                treeTranslation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH])) 

            
            for i in range(len(self.meshes)):
                
                # rotation = self.getMeshRotation(self.meshRotations[i], cameraRotation) 
                # translation = self.getMeshTranslation(self.meshTranslations[i], treeTranslation)
                dx = self.meshTranslations[i][0]
                dy = self.meshTranslations[i][1] - self.TREE_DY # NEed to compensate for the move up. We don't want the branch translated that far. 
                dz = self.meshTranslations[i][2]
                
                branchTranslation = np.transpose(mt.create_from_translation([dx, dy, dz]))
                # # negBranchTranslation = np.transpose(mt.create_from_translation(-1 * np.array(self.meshTranslations[self.index])))
                xRad = self.angle_to_radians(self.meshRotations[i][0])
                yRad = self.angle_to_radians(self.meshRotations[i][1])
                zRad = self.angle_to_radians(self.meshRotations[i][2])
                branchRotation = mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) 

                scale = mt.create_from_scale(self.meshScales[i]) # get the scale at that index [0.1, 0.1, 0.1]
                
                model = mt.create_identity()
                # Need the branch to stay on the tree and then rotate with it
                # model = translation @ rotation @ scale # translation @ rotation @ scale
                model = treeTranslation @ cameraRotation @ branchTranslation @ branchRotation @ scale
                # SET SHADER PROGRAM 
                modelLoc = gl.glGetUniformLocation(self.program, "model")
                gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation

           
                gl.glBindVertexArray(self.VAOs[i])
                vertices = np.array(self.meshes[i].vertices, dtype=np.float32)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))
                self.currentFeature = self.meshDescriptions[i]
                # print(f"Current feature is {self.currentFeature}")                    

                gl.glBindVertexArray(0) # unbind the vao
            
            # ADD LABELS:
            binLabelLocs = [(450, 250), (600, 215), (750, 200)] # 
            for i in range(len(self.meshes)):
                branch_label = f"Branch {i+1}"
                x, y = binLabelLocs[i]
                scale = mt.create_from_scale(self.meshScales[i]) # get the scale at that index [0.1, 0.1, 0.1]
                # x, y = self.calculateLabelPlacement(position=self.meshTranslations[i],     # how much we needed to move the object by
                #                                     translation=treeTranslation,           # where in the tree
                #                                     rotation=cameraRotation,               # just where is the camera location
                #                                     scale=scale)              # how much to scale the branch by
                _ = self.renderText(branch_label, x, y, 0.4, color=[0, 1, 0])


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


    def renderText(self, text, x, y, scale, color=None):
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # move screen pose if it is outside the view
        if x > self.width:
            x = x - (x - self.width)
        
        if y > self.height:
            y = y - (y - self.height) 


        if color is None:
            color = [1, 1, 0]

        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()
        gl.glUseProgram(self.textProgram) # activate the program
        # allows us to map the text on screen using screen coordinates
        textProject = np.transpose(mt.create_orthogonal_projection_matrix(0.0, self.width, 0.0, self.height, self.ZNEAR, self.ZFAR))
        textProjLoc = gl.glGetUniformLocation(self.textProgram, "projection")
        gl.glUniformMatrix4fv(textProjLoc, 1, gl.GL_TRUE, textProject)

        colorLoc = gl.glGetUniformLocation(self.textProgram, "textColor")
        gl.glUniform3fv(colorLoc, 1, color)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self.textVAO)
        # want to get the bounding box of the text
        
        minX = x
        minY = y
        maxX = x
        maxY = y
        for char in text:
            # intChar = ord(char) # convert the character to a number
            character = self.characters[char]

            xpos = x + character.bearing[0] * scale # get the x value from the bearing
            ypos = y - (character.size[1] - character.bearing[1]) * scale # get the y value from bearing and scale

            w = character.size[0] * scale
            h = character.size[1] * scale

            maxX = xpos + w
            maxY = np.max([ypos + h, maxY])

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
        gl.glEnable(gl.GL_DEPTH_TEST)
        # gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_BLEND)
        bounds = [minX, maxX, self.height - minY, self.height - maxY]
        return bounds
         

    def initializeLabels(self):
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
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, self.labelLines.nbytes, self.labelLines, gl.GL_DYNAMIC_DRAW)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.testCube.nbytes, self.testCube, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * self.testCube.itemsize, ctypes.c_void_p(0))
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)



    def getLabelPoints(self, screenPose):
        toUpdate = False # do I need to send an update to the system to redraw
        for i, label in enumerate(self.jsonData["Features"]):
            # print(f"Label {label}: {screenPose[i]}")
            x, y = screenPose[i]
            start = 6 * i
            end = 6 * (i+1)
            # projection on the screen
            # textProject = np.transpose(mt.create_orthogonal_projection_matrix(0.0, self.width, 0.0, self.height, self.ZNEAR, self.ZFAR))
            # print(textProject)
            # text uses a different projection to 

            # find the position on the screen in local coordinates
            u,v = self.convertXYtoUV(x, y)  # IS IT GETTING THE FULL WIDTH X HEIGHT for the 
            # print(f"[{u}, {v}, d=0]")
           
            # 
            # xyz_w = inv_mvp @ np.transpose([u, v, 0, 1])
            # xyz_w = inv_mvp @ np.transpose([u, v, 0.1, 1])
            xyz_w = self.convertUVDtoXYZ(u=u, v=v, d=0.1)
            # print(f"Label {label}: {xyz_w}")
            
            if not np.all(np.isclose(self.labelLines[start:start+3], xyz_w[:3])):
                print("Label Point Different")
                print(f"Label Lines: {self.labelLines[start:start+3]}")
                print(f"xyz_w: {xyz_w[:3]}")
                toUpdate = True
                self.labelLines[start:start+3] = xyz_w[:3] # self.convertUVDtoXYZ(u, v, 0)[:3] # want at the 0 position on the screen

            # convert the point from label features back from 
            # Convert from Blender coordinate system (+x right, +y into screen, +z up) to OpenGL coordinate system (+x right, +y up, +z out of screen)
            # -90 degree rotation around the x axis
            
            endPt = [self.jsonData["Features"][label][0], self.jsonData["Features"][label][2], -1 * self.jsonData["Features"][label][1]] # +y --> -z 
            # print(f"uvd={endPt} conversion: {xyz_w}")
            self.labelLines[end-3:end] = np.array(endPt, dtype=np.float32) # locations stored in local space
        
        if toUpdate:
            print("Update Label Lines")
            gl.glNamedBufferSubData(self.labelVBO, 0, self.labelLines.nbytes, self.labelLines)
            self.update()
        else:
            print("Don't Update Label Lines")
            


    def drawLabels(self, screenPose):
        # Loop through each label
        # Need to convert x, y of screen positions to local space coordinates
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()
        
        # self.labelLines = np.zeros( 4 * 2 * 3 ) # 4 labels * 2 pts per label * 3 dimensions
        # print(self.projection @ self.view @ self.model)
        mvp = self.projection @ self.view @ self.model
        inv_mvp = np.linalg.inv(mvp)
        
        for i, label in enumerate(self.jsonData["Features"]):
            # print(f"Label {label}: {screenPose[i]}")
            x, y = screenPose[i]
            
            # find the position on the screen in local coordinates
            u,v = self.convertXYtoUV(x, y)  # IS IT GETTING THE FULL WIDTH X HEIGHT for the 
            # print(f"[{u}, {v}, d=0]")
            mvp = self.projection @ self.view @ self.model
            
            xyz_w = self.convertUVDtoXYZ(u=u, v=v, d=0.1)
            # print(xyz_w)

        # test = mvp @ np.transpose([-0.5, 2, 0, 1])
        # test /= test[3]
        # print(test)


        # self.getLabelPoints(screenPose) # get where to draw and update (if necessary)
        
        # render text on the screen
        for i, label in enumerate(self.jsonData["Features"]):
            x, y = screenPose[i]
            # print(f"Label {label}: {screenPose[i]}")
            _ = self.renderText(label, x, y, 0.5)
        
        gl.glUseProgram(0)
        gl.glUseProgram(self.labelProgram)

        modelLoc = gl.glGetUniformLocation(self.labelProgram, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, self.model) # self.rotation

        projLoc = gl.glGetUniformLocation(self.labelProgram, "projection")
        gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection) # use the same projection values

        viewLoc = gl.glGetUniformLocation(self.labelProgram, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) # use the same location view values

        colorLoc = gl.glGetUniformLocation(self.labelProgram, "color")
        gl.glUniform3fv(colorLoc, 1, [1, 0, 0])

        # BIND VAO AND TEXTURE
        gl.glBindVertexArray(self.labelVAO)
        # gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.skyTexture)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.labelVBO)
        
        # gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, lines.nbytes, lines)

        # gl.glDrawArrays(gl.GL_LINES, 0, int(self.labelLines.size)) 
        # gl.glDrawArrays(gl.GL_LINES, 0, int(self.testCube.size))
        
        gl.glUseProgram(0)
        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()




    def drawSkyBox(self):
        gl.glLoadIdentity()
        gl.glPushMatrix()
        gl.glUseProgram(0)
        # gl.glDisable(gl.GL_CULL_FACE)
       
        gl.glDisable(gl.GL_DEPTH_TEST)
        # set the depth function to equal
        oldDepthFunc = gl.glGetIntegerv(gl.GL_DEPTH_FUNC)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_FALSE)


        # Deal with the rotation of the object
        # scale = mt.create_from_scale([-4.3444, 4.1425, -10.00])
        scale = mt.create_from_scale([-4, 4, -9.99]) # -4, 4, -9.99
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
        gl.glEnable(gl.GL_DEPTH_TEST)

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

        stride = self.boundBoxVertices.itemsize * 3
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

        colorLoc = gl.glGetUniformLocation(self.boundBoxProgram, "color")
        gl.glUniform3fv(colorLoc, 1, [1, 0, 0])

        # BIND VAO AND TEXTURE
        gl.glBindVertexArray(self.boundBoxVAO)
        # TO FIX
        gl.glLineWidth(2.0)
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, int(self.boundBoxVertices.size / 3))  # draw the vertices of the skybox 
        # gl.glDrawElements(gl.GL_QUADS, int(self.skyVertices.size), gl.GL_UNSIGNED_INT, 0)

        gl.glUseProgram(0)
        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()


    def initializeScalePruneDrawing(self):
        gl.glUseProgram(0)

        vertexShader = Shader("vertex", "simple_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "simple_shader.frag").shader

        self.pruneProgram = gl.glCreateProgram()
        gl.glAttachShader(self.pruneProgram, vertexShader)
        gl.glAttachShader(self.pruneProgram, fragmentShader)
        gl.glLinkProgram(self.pruneProgram)

        gl.glUseProgram(self.pruneProgram)

        self.pruneVAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.pruneVAO)

        self.pruneVBO = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pruneVBO)

        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pruneLines.nbytes, self.drawVertices, gl.GL_DYNAMIC_DRAW) # GL_STATIC_DRAW

        stride = self.pruneLines.itemsize * 3

        # enable the pointer for the shader
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))  

    
    def drawScalePruningLines(self):
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        gl.glUseProgram(self.pruneProgram) 
        if self.toPrune:
            # SET THE LIGHT COLOR
            # treeTranslation = np.transpose(mt.create_from_translation([0, 0, self.TREE_SECTION_DEPTH])) # self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH
            if self.wholeView:
                treeTranslation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH]))

            cutTranslation = np.transpose(mt.create_from_translation([0, 0, 0]))
            # Compensate for different types of prunes
            x = self.meshTranslations[self.index][0] # -0.1, x value should be the same
            y = self.meshTranslations[self.index][1] - self.TREE_DY # 0.18
            z = self.meshTranslations[self.index][2] # 0 same z value
            
            color = [1, 0, 0]

            # Need to change the position of the pruning cut depending on the rotation
            # If x rotated 90 degrees --> z direction 
            # If rotation in the z direction --> x direction 
            xRad = self.angle_to_radians(self.meshRotations[self.index][0])
            yRad = self.angle_to_radians(self.meshRotations[self.index][1])
            zRad = self.angle_to_radians(self.meshRotations[self.index][2])
            branchRotation = mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) 


            BRANCH = 0.15 # how much you need to translate branches to sit on the secondary branch next to the trunk

            if self.pruneDescription[self.index] == "Heading Cut":
                # translate = self.meshTranslations[self.index]
                        
                # adjust for where the heading cut is based on the location of the secondary branch
                # How much to translate in the y direction
                if xRad == 0:
                    dy = (0.175 - self.meshTranslations[self.index][1] + (BRANCH - self.meshTranslations[self.index][1])) # *math.sin(xRad)
                else:
                    dy = (0.175 - self.meshTranslations[self.index][1] - (BRANCH - self.meshTranslations[self.index][1])*math.sin(xRad))

                cut_pos = [0, dy, 0, 1] 
                dx, dy, dz, _ = branchRotation @ np.transpose(cut_pos)

                # need it to be the same y direction 
                # dy += 0.2 - self.meshTranslations[self.index][1] # correct translation to sit at 0.2
                # dy -= 0.18 # CAN CHANGE THE VALUE
                # cutTranslation = np.transpose(mt.create_from_translation([-0.1, 0.18, 0])) # translate to a certain distance away from the branch
            else:
                # translate = self.meshTranslations[self.index]
                # dy += 0.1 - self.meshTranslations[self.index][1] # correct translation to sit at 0.1
                # dy -= 0.1 # 0.1
                # if xRad == 0:
                #     dy = (0.08 - self.meshTranslations[self.index][1] + (BRANCH - self.meshTranslations[self.index][1])) # *math.sin(xRad)
                # else:
                #     dy = (0.08 - self.meshTranslations[self.index][1] - (BRANCH - self.meshTranslations[self.index][1])*math.sin(xRad))

                cut_pos = [0, 0.08 - self.meshTranslations[self.index][1], 0, 1]
                color = [0, 1, 0]
                dx, dy, dz, _ = branchRotation @ np.transpose(cut_pos)

                # print(f"{x} / math.tan({zRad}) = {x} / {math.tan(zRad)} = {z / math.tan(zRad)}")
                dx -= ((z * math.tan(np.pi - zRad)) + 0.01)
                # dx += (1.4*z / math.tan(zRad)) # 0.75*z
                # dx += x * math.tan(xRad)
                dz += 0.03 * math.tan(np.pi - zRad)
                # dz -= (0.075 / math.tan(zRad)) # Want around -0.05 for z when angled in 90 degree rotation


            cutTranslation = np.transpose(mt.create_from_translation([x + dx, y + dy, z + dz]))

            # translation = self.getMeshTranslation(translate, treeTranslation)
            # translation = np.transpose(mt.create_from_translation(self.meshTranslations[self.index]))

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle) # camera rotations

           

            scale = mt.create_from_scale([0.025, 0.005, 0.025]) # Scale the box to the correct size
    
            # Need to translate the value by the same amount as the branch
            # model = translation @ rotation @ scale
            # model = translate @ treeTranslation @ rotation @ scale

            model = treeTranslation @ cameraRotation @ cutTranslation @ branchRotation @ scale

            # print("Whole View", self.wholeView)
            # print("MVP Matrix:\n", self.projection @ self.view @ model)
            
            modelLoc = gl.glGetUniformLocation(self.pruneProgram, "model")
            gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation

            projLoc = gl.glGetUniformLocation(self.pruneProgram, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)

            viewLoc = gl.glGetUniformLocation(self.pruneProgram, "view")
            gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

            colorLoc = gl.glGetUniformLocation(self.pruneProgram, "color")
            gl.glUniform3fv(colorLoc, 1, color)

            # # self.draw(self.vertices)
            # end = self.drawCount * 3
            gl.glBindVertexArray(self.pruneVAO)
            # gl.glLineWidth(2.0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pruneVBO)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pruneLines.nbytes, self.pruneLines, gl.GL_DYNAMIC_DRAW)

            # gl.glPointSize(3.0)
            gl.glLineWidth(5.0)
            if self.pruneDescription[self.index] != "Don't Prune":
                gl.glDrawArrays(gl.GL_QUADS, 0, int(self.pruneLines.size / 3))
            # gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.drawCount) 

        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0)



    # DESCRIPTION: Draw branches as "bins" that have a specific feature.
    #   - connected to the vigor terminology interface interaction
    def drawBinFeatures(self):
                        
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        print("Inside Bin Features")
        drawing = False
        if len(self.meshes) > 0: # check I have values just in case
            # Bind the texture and link the program
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.branchTexture)
            gl.glUseProgram(self.branchProgram) 

            # BIND VALUES THAT DON'T CHANGE WITH THE MANIPULATION
            lightPosLoc = gl.glGetUniformLocation(self.program, "lightPos")
            gl.glUniform3fv(lightPosLoc, 1, self.lightPos)
            lightColorLoc = gl.glGetUniformLocation(self.program, "lightColor")
            gl.glUniform3fv(lightColorLoc, 1, self.lightColor)
            treeColorLoc = gl.glGetUniformLocation(self.program, "color")
            gl.glUniform3fv(treeColorLoc, 1, self.tree_color_dark)

            projLoc = gl.glGetUniformLocation(self.program, "projection")
            gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)
            viewLoc = gl.glGetUniformLocation(self.program, "view")
            gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

            hAngle = self.angle_to_radians(self.turntable)
            vAngle = self.angle_to_radians(self.vertical)
            # rotation
            cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle)

            if self.wholeView: # looking at the whole tree view
                treeTranslation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH])) 

            
            for i in range(len(self.meshes)):
                
                # rotation = self.getMeshRotation(self.meshRotations[i], cameraRotation) 
                # translation = self.getMeshTranslation(self.meshTranslations[i], treeTranslation)
                dx = self.meshTranslations[i][0]
                dy = self.meshTranslations[i][1] - self.TREE_DY # Need to compensate for the move up. We don't want the branch translated that far. 
                dz = self.meshTranslations[i][2]
                
                branchTranslation = np.transpose(mt.create_from_translation([dx, dy, dz]))
                # # negBranchTranslation = np.transpose(mt.create_from_translation(-1 * np.array(self.meshTranslations[self.index])))
                xRad = self.angle_to_radians(self.meshRotations[i][0])
                yRad = self.angle_to_radians(self.meshRotations[i][1])
                zRad = self.angle_to_radians(self.meshRotations[i][2])
                branchRotation = mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) 

                scale = mt.create_from_scale(self.meshScales[i]) # get the scale at that index [0.1, 0.1, 0.1]
                
                model = mt.create_identity()
                # Need the branch to stay on the tree and then rotate with it
                model = treeTranslation @ cameraRotation @ branchTranslation @ branchRotation @ scale
                # SET SHADER PROGRAM 
                modelLoc = gl.glGetUniformLocation(self.program, "model")
                gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation

                print(f"wanted feature: {self.wantedFeature}")
                if self.wantedFeature == self.meshDescriptions[i]:
                    # print(f"Drawing Bin with feature: {self.meshDescriptions[i]}\n")
                    gl.glBindVertexArray(self.VAOs[i])
                    vertices = np.array(self.meshes[i].vertices, dtype=np.float32)
                    gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(vertices.size / 3))
                    self.currentFeature = self.meshDescriptions[i]                    

                    gl.glBindVertexArray(0) # unbind the vao
                    # ADD LABELS
                    drawing = True
                    
                
                gl.glBindVertexArray(0) # unbind the vao
            # END FOR
            if drawing:
                binLabelLocs = [(350, 250), (500, 215), (650, 200)] # (1100, 650), (1450, 600), (1800, 550)
                for i in range(3):
                    branch_label = f"{self.wantedFeature} Branch {i+1}"
                    x, y = binLabelLocs[i]
                    scale = mt.create_from_scale(self.meshScales[i]) # get the scale at that index [0.1, 0.1, 0.1]
                    _ = self.renderText(branch_label, x, y, 0.3)
        # END IF

        gl.glPopMatrix()
        gl.glUseProgram(0)







    def initializeBinPruneDrawing(self):
        gl.glUseProgram(0)

        if len(self.meshes) > 0:
            vertexShader = Shader("vertex", "simple_shader.vert").shader # get out the shader value    
            fragmentShader = Shader("fragment", "simple_shader.frag").shader

            self.multiPruneProgram = gl.glCreateProgram()
            gl.glAttachShader(self.multiPruneProgram, vertexShader)
            gl.glAttachShader(self.multiPruneProgram, fragmentShader)
            gl.glLinkProgram(self.multiPruneProgram)

            gl.glUseProgram(self.multiPruneProgram)


            for i, _ in enumerate(self.meshes): # len(files)

                    # create and bind VAOs and VBOs
                    self.pruneVAOs[i] = gl.glGenVertexArrays(1)
                    gl.glBindVertexArray(self.pruneVAOs[i])

                    self.pruneVBOs[i] = gl.glGenBuffers(1)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pruneVBOs[i])
                    
                    # Load in the Mesh file and vertex
                    # fname = "../obj_files/" + self.manipulation + "/" + str(files[i]["Name"])
                    # self.meshes[i] = Mesh(fname)
                    # vertices = np.array(mesh.vertices, dtype=np.float32) # self.meshes[i]

                    gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pruneLines.nbytes, self.pruneLines, gl.GL_DYNAMIC_DRAW) # Allows me to grow the number of pruning lines in the program
                    # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex.nbytes, self.vertex, gl.GL_STATIC_DRAW)

                    # SET THE ATTRIBUTE POINTERS SO IT KNOWS LCOATIONS FOR THE SHADER
                    
                    stride = self.pruneLines.itemsize * 3

                    # enable the pointer for the shader
                    gl.glEnableVertexAttribArray(0)
                    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))

                    # Reset the buffers 
                    gl.glBindVertexArray(0)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
  

    
    def drawBinPruningLines(self):
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()

        gl.glUseProgram(self.multiPruneProgram) 
        if self.toPruneBin:
            # SET THE LIGHT COLOR
            # treeTranslation = np.transpose(mt.create_from_translation([0, 0, self.TREE_SECTION_DEPTH])) # self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH
            if self.wholeView:
                treeTranslation = np.transpose(mt.create_from_translation([0, self.TREE_DY, self.WHOLE_TREE_DEPTH]))
            else:
                treeTranslation = np.transpose(mt.create_from_translation([self.TREE_SECTION_DX, self.TREE_DY, self.TREE_SECTION_DEPTH]))

            
            for i in range(len(self.meshes)):
                
                if self.pruneBranches[i] != "Don't Prune":
                    # Compensate for different types of prunes
                    x = self.meshTranslations[i][0] # -0.1, x value should be the same
                    y = self.meshTranslations[i][1] - self.TREE_DY # 0.18
                    z = self.meshTranslations[i][2] # 0 same z value
                    
                    color = [1, 0, 0]

                    # Need to change the position of the pruning cut depending on the rotation
                    # If x rotated 90 degrees --> z direction 
                    # If rotation in the z direction --> x direction 
                    xRad = self.angle_to_radians(self.meshRotations[i][0])
                    yRad = self.angle_to_radians(self.meshRotations[i][1])
                    zRad = self.angle_to_radians(self.meshRotations[i][2])
                    branchRotation = mt.create_from_x_rotation(xRad) @ mt.create_from_y_rotation(yRad) @ mt.create_from_z_rotation(zRad) 


                    if self.pruneDescription[i] == "Heading Cut":
                        # translate = self.meshTranslations[self.index]
                        BRANCH = 0.15 # how much you need to translate branches to sit on the secondary branch next to the trunk
                        
                        # adjust for where the heading cut is based on the location of the secondary branch
                        # How much to translate in the y direction
                        if xRad == 0:
                            dy = (0.175 - self.meshTranslations[i][1] + (BRANCH - self.meshTranslations[i][1])) # *math.sin(xRad)
                        else:
                            dy = (0.175 - self.meshTranslations[i][1] - (BRANCH - self.meshTranslations[i][1])*math.sin(xRad))
                        # ((self.meshTranslations[i][1] - BRANCH)/math.cos(xRad))
                        # print(f"Branch {i+1}: {dy}")

                        cut_pos = [0, dy, 0, 1]
                        dx, dy, dz, _ = branchRotation @ np.transpose(cut_pos)

                    else:
                        cut_pos = [0, 0.08 - self.meshTranslations[i][1], 0, 1]
                        color = [0, 1, 0]
                        dx, dy, dz, _ = branchRotation @ np.transpose(cut_pos)

                        dx -= ((z * math.tan(np.pi - zRad)) + 0.01)
                        dz += 0.03 * math.tan(np.pi - zRad)
                    
                    # print(f"Branch {i}: {cut_pos}") 
                    cutTranslation = np.transpose(mt.create_from_translation([x + dx, y + dy, z + dz]))
                    # print(f"Cut Translation Branch {i+1}: {[x + dx, y + dy, z + dz]}")

                    hAngle = self.angle_to_radians(self.turntable)
                    vAngle = self.angle_to_radians(self.vertical)
                    cameraRotation = mt.create_from_y_rotation(hAngle) @ mt.create_from_x_rotation(vAngle) # camera rotations

                    scale = mt.create_from_scale([0.025, 0.005, 0.025]) # Scale the box to the correct size
        
                    # Need to translate the value by the same amount as the branch
                    # model = translation @ rotation @ scale
                    # model = translate @ treeTranslation @ rotation @ scale

                    model = treeTranslation @ cameraRotation @ cutTranslation @ branchRotation @ scale
                    
                    modelLoc = gl.glGetUniformLocation(self.multiPruneProgram, "model")
                    gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation

                    projLoc = gl.glGetUniformLocation(self.multiPruneProgram, "projection")
                    gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)

                    viewLoc = gl.glGetUniformLocation(self.multiPruneProgram, "view")
                    gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) 

                    colorLoc = gl.glGetUniformLocation(self.multiPruneProgram, "color")
                    gl.glUniform3fv(colorLoc, 1, color)

                    # # self.draw(self.vertices)
                    # end = self.drawCount * 3
                    gl.glBindVertexArray(self.pruneVAOs[i])
                    # gl.glLineWidth(2.0)
                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pruneVBOs[i])
                    gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pruneLines.nbytes, self.pruneLines, gl.GL_DYNAMIC_DRAW)

                    # gl.glPointSize(3.0)
                    gl.glLineWidth(5.0)
                    gl.glDrawArrays(gl.GL_QUADS, 0, int(self.pruneLines.size / 3))

        gl.glBindVertexArray(0) # unbind the vao
        gl.glPopMatrix()
        gl.glUseProgram(0)





    def initializeDrawing(self):
        gl.glUseProgram(0)

        vertexShader = Shader("vertex", "draw_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "draw_shader.frag").shader

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

        stride = self.drawVertices.itemsize * 6 # 3 for pos 3 for color

        # enable the pointer for the shader
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))  

        # Add a new vertex attribute that correlates with color
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12)) # Start at pos 3 with float size 4 --> 3*4

    
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
        
        colorLoc = gl.glGetUniformLocation(self.drawProgram, "color")
        gl.glUniform3fv(colorLoc, 1, [1, 0, 0])

        # # self.draw(self.vertices)
        # end = self.drawCount * 3
        gl.glBindVertexArray(self.drawVAO)
        # gl.glLineWidth(2.0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.drawVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.drawVertices.nbytes, self.drawVertices, gl.GL_DYNAMIC_DRAW)

        # gl.glPointSize(3.0)
        gl.glLineWidth(5.0)
        gl.glDrawArrays(gl.GL_QUADS, 0, int(self.drawVertices.size / 6)) # 6 given 3 vertices 3 color
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.drawCount) 


        if len(self.cutSequenceDict["Rule"]) == 0:
            self.endCutTime = time.time()
        

        # Need to determine if I display the text of the pruning choices
        if self.crossYMenu:
            self.ruleTimeStart = time.time() # Start recording time between decisions
            # print("Display text on the screen")
            # print(f"{self.lastPose.x()}, {self.lastPose.y()}")
            lastX, lastY =  (self.lastPose.x()+10, self.lastPose.y())
            # display coordinates in text coordinates (0 in bottom left)
            crossYPos = [(lastX, self.height - lastY), (lastX, self.height - lastY - 30), (lastX, self.height - lastY - 60)]
            text = ["Branch Vigor", "Canopy Cut", "Bud Spacing"]
            colors = [self.vigorColor, self.canopyColor, self.spacingColor]
            for i in range(len(text)):
                x, y = crossYPos[i]
                # print(f"{text[i]}: {x}, {y}")
                self.crossyTextBounds[text[i]] = self.renderText(text[i], x=x, y=y, scale=0.5, color=colors[i])
                # self.crossyTextBounds[text[i]][2] = -1 * (self.crossyTextBounds[text[i]][2] - self.height)  # fix to back to screen coordinates and not text coordinates   
                # self.crossyTextBounds[text[i]][3] = -1 * (self.crossyTextBounds[text[i]][3] - self.height)
                # print(f"{text[i]} bounds: {self.crossyTextBounds[text[i]]}")


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

        treeColorLoc = gl.glGetUniformLocation(self.program, "color")
        gl.glUniform3fv(treeColorLoc, 1, self.tree_color_light)

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

        if self.screenType == "draw_tutorial":
            _ = self.renderText("A", x=200, y=250, scale=1, color=[1, 1, 0]) # 600, 550
            _ = self.renderText("B", x=300, y=200, scale=1, color=[1, 1, 0]) # 600, 550


       

    def initializeGL(self):
        # print(self.getGlInfo())
        # gl.glClearColor(0, 0.224, 0.435, 1)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        # gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        # gl.glClearColor(0.56, 0.835, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND) # for text and skybox

        # gl.glEnable(gl.GL_CULL_FACE)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glDisable(gl.GL_CULL_FACE)
        # gl.glEnable(gl.GL_CULL_FACE)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
         
        self.initializeTreeMesh()

        self.initializeLabels()

        self.initializeManipulationFiles() # Also for the binning values

        self.initializeBinPruneDrawing()

        self.initializeScalePruneDrawing()

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
        gl.glUseProgram(0)

        #Paint the skybox regardless of the view
        self.drawSkyBox()
        gl.glUseProgram(0) 

        # Paint the tree regardless of the view
        self.paintTree()  

        if self.wholeView:
            self.drawBoundingBox()

        if not self.wholeView and self.displayLabels:
            screenPose = [(350, 300), # trunk (950, 300)
                          (50, 400), # Secondary (250, 650)
                          (820, 250), # Tertiary branch (2200, 600)
                          (650, 95)] # bud TO CHANGE (2000, 1000)
            self.drawLabels(screenPose)
            
        if self.toManipulate: #  and not self.wholeView
            self.drawManipulationBranch()

        if self.toBin: #  and not self.wholeView
            self.drawBinBranches()
        
        if self.toBinFeatures:
            self.drawBinFeatures()

        if self.toPruneBin and not self.wholeView:
            self.drawBinPruningLines()

        if self.termDraw:
            self.drawTermSections()

        if self.toPrune and not self.wholeView: # Draw the pruning lines on top of the tree
            self.drawScalePruningLines()

        # WANT TO DRAW THE POINTS BASED ON WHAT SOMEONE WAS DRAWING
        if self.drawLines and not self.wholeView:
            self.drawPruningLines()

            
    

    # MOUSE ACTION EVENTS WITH GLWIDGET SCREEN
    def mousePressEvent(self, event) -> None:
        # event.pos()
        # p = event.globalPosition().toPoint()
        self.press = QPoint(event.pos()) # returns the last position of the mouse when clicked
        self.startPose = QPoint(event.pos()) # returns the last position of the mouse when clicked
        self.pressTime = time.time()
    

    def mouseReleaseEvent(self, event) -> None:
        self.release = QPoint(event.pos()) 

        self.lastPose = QPoint(event.pos()) # event.pos()
        self.releaseTime = time.time()
        if abs(self.lastPose.x() - self.startPose.x()) > 5 or abs(self.lastPose.y() - self.startPose.y()) > 5: # and
            self.rayDraw()
        else:
            # print(f"Crossy Menu on Click: {self.crossYMenu}")
            if self.crossYMenu:
                self.crossYDecision()
            else:
                print("Do Nothing")        
        # _ = self.rayDirection(self.lastPose.x(), self.lastPose.y())
        # self.update()

   


    def updateCutColorToDecision(self):
        if self.drawCount > 0:
            # 8 vertices
            # 3 pos + 3 color
            # 8 * 6 = 48
            stride = 8 * 6 * 3 # 8 vertices (cube) with 6 points (each vertex having pos (3) and color (3)) each with points a length of 3
            start = (self.drawCount - stride)  # (self.drawCount - 24) * 3
            # need to replace values at that section in draw vertices to null
            # loop over the range from start to end and replace every values with index 3-5 with pen color
            for i in range(int(stride / 6)): # divided by 6 as we take every 6 section chunk
                idx = (start + 3) + (6 * i) 
                self.drawVertices[idx:idx+3] = self.penColor
            
            gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)
        
        self.update() # update the GL Call
        

    def crossYDecision(self):
        # self.crossYMenu = False
        # Look at position of the drawing strokes 
        self.penColor = [1, 0, 0]

        startX = self.startPose.x()
        startY = self.startPose.y()
        endX = self.lastPose.x()
        endY = self.lastPose.y()

        # print(f"({startX}, {startY}) to ({endX}, {endY})")
        
        # check if the start value is in the text or if the end value is in the 
        for decision in self.crossyTextBounds:
            bound = self.crossyTextBounds[decision]
            # print(f"{decision}:")
            intersect = self.clickInBounds(start=(startX, startY), end=(endX, endY), bounds=bound)
            # intersect = self.lineInTextBounds(start=(startX, startY), end=(endX, endY), bounds=bound)

            if intersect:
                # print("Intersect detected!")
                self.setPenColor(decision=decision)
                self.crossYMenu = False # only reset the crossYMenue when you have successfully set the decision to a new value
                
                self.endCutTime = time.time()

                ruleTime = self.endCutTime - self.ruleTimeStart # Get the end of time between cuts
                print(f"Rule Time: {ruleTime}")
                self.cutSequenceDict["Rule Time"].append(ruleTime)

                self.updateCutColorToDecision() # update the color of the drawing and go!
                break
        # print(f"CrossYMenu decision: {self.crossYMenu}")
    
    
    def clickInBounds(self, start, end, bounds):
        sX, sY = start
        eX, eY = end
        minX = np.min([bounds[0], bounds[1]])
        maxX = np.max([bounds[0], bounds[1]])
        minY = np.min([bounds[2], bounds[3]])
        maxY = np.max([bounds[2], bounds[3]])
       
        return eX >= minX and eX <= maxX and eY >= minY and eY <= maxY



    # def lineInTextBounds(self, start, end, bounds):
    #     # check if the values are in the bounds
    #     # Bounds is a list of floats from [minX, maxX, minY, maxY] of the text

    #     # IF STARTING AND END POINTS DON'T INTERSECT TO START
    #     # shift everything to make it appear the start point is x=0 and end point y = 0
    #     sX = 0
    #     sY = start[1] - end[1]

    #     eX = end[0] - start[0]
    #     eY = 0

    #     print(f"({sX}, {sY}) to ({eX}, {eY})")

    #     # get the bounds in the shifted space
    #     minX = bounds[0] - start[0]
    #     maxX = bounds[1] - start[0]
    #     minY = bounds[2] - end[1]
    #     maxY = bounds[3] - end[1]

    #     print(f"Bounds: ({minX}, {maxX}, {minY}, {maxY})")

    #     # line completely encompassed in the bounds
    #     if sX >= minX and sX <= maxX and sY >= minY and sY <= maxY:
    #         return True
    #     elif eX >= minX and eX <= maxX and eY >= minY and eY <= maxY:
    #         return True

    #     # get the slope of the line
    #     if eX - sX > 0:
    #         slope = (eY - sY) / (eX - sX)
    #     else:
    #         slope = (eY - sY) / 1e-13 # ensure no divide by 0

    #     # looking for the values in the slope to see if it intersects the square before it exits the region between min and max X
    #     yStart = slope * minX + sY
    #     print(f"Intersect value minX: {yStart}")
    #     # check if the point is inside the bounds of 
    #     if yStart >= minY and yStart <=maxY:
    #         return True
        
    #     yEnd = slope * maxX + sY 
    #     print(f"Intersect value maxX: {yEnd}")
    #     if yEnd >= minY and yEnd <=maxY:
    #         return True

    #     else:
    #         return False 

    
    

    def convertXYtoUV(self, x=0, y=0):
        # print(f"convertXYtoUV -- Width: {self.width} x Height: {self.height}")
        u = ((2 * x) / self.width) - 1.0 
        v = 1.0 - ((2 * y) / self.height)
        # print(u, v)
        return u, v


    def convertUVtoScreenCoords(self, u=0, v=0):
        x = ((u + 1) * self.width) / 2
        y = ((1.0 - v) * self.height) / 2
        return x, y

    
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
            start = (self.drawCount) + 6 * i 
            # localPt = self.convertWorldToLocal(quad[i])
            self.drawVertices[start:start+3] = drawPts[i]
            self.drawVertices[start+3:start+6] = [1, 0, 0] #self.penColor

        self.drawCount += len(drawPts) * 2 * 3 # add 24 pts * 2 (color and vertices) * 3 (length of array)


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

        drawPt1 = self.get_drawn_coords(u1, v1, minZ) # returns a length 3 array
        drawPt2 = self.get_drawn_coords(u1, v1, maxZ)
        drawPt3 = self.get_drawn_coords(u2, v2, maxZ)
        drawPt4 = self.get_drawn_coords(u2, v2, minZ)
        drawPt5 = self.get_drawn_coords(u3, v3, minZ)
        drawPt6 = self.get_drawn_coords(u3, v3, maxZ)
        drawPt7 = self.get_drawn_coords(u4, v4, minZ)
        drawPt8 = self.get_drawn_coords(u4, v4, maxZ)

        vertices = [drawPt1, drawPt2, drawPt3, drawPt4, drawPt5, drawPt6, drawPt7, drawPt8]

        # cubeVertices = [drawPt1, drawPt2, drawPt3, drawPt4,
        #                 drawPt1, drawPt2, drawPt6, drawPt5, 
        #                 drawPt5, drawPt6, drawPt7, drawPt8,
        #                 drawPt3, drawPt4, drawPt7, drawPt8,
        #                 drawPt2, drawPt4, drawPt6, drawPt8,
        #                 drawPt1, drawPt3, drawPt5, drawPt7]
        cubeVertices = [drawPt1, drawPt2, drawPt3, drawPt4,
                        drawPt5, drawPt8, drawPt7, drawPt6, 
                        drawPt1, drawPt5, drawPt6, drawPt2,
                        drawPt2, drawPt6, drawPt7, drawPt3,
                        drawPt3, drawPt7, drawPt8, drawPt4,
                        drawPt5, drawPt1, drawPt4, drawPt8]

        return cubeVertices, vertices 




    def rayDraw(self):
        # Use the first and last pose to get the midpoint
        
        # checking the midpoint value
        # midPt = [(self.startPose.x() + self.lastPose.x())/2, (self.startPose.y() + self.lastPose.y())/2]
        # dir = self.rayDirection(x=midPt[0], y=midPt[1])[:3]
        # print(dir)
        if not self.crossYMenu:
            u1, v1 = self.convertXYtoUV(x=self.startPose.x(), y=self.startPose.y())
            u2, v2 = self.convertXYtoUV(x=self.lastPose.x(), y=self.lastPose.y())

            # print(f"Start ({self.startPose.x()}, {self.startPose.y()}), or ({u1}, {v1})")
            # print(f"End ({self.lastPose.x()}, {self.lastPose.y()}) or ({u2}, {v2})")
            # returns intersect faces in local coordinates
            intersectFaces = self.mesh.intersect_faces(u1=u1, v1=v1, u2=u2, v2=v2, projection=self.projection, view=self.view, model=self.model)
            # intersectFaces = None
            if intersectFaces is not None:
                # Determine faces in a given region
                self.cutReleaseTime = self.releaseTime
                cutTime = self.cutReleaseTime - self.pressTime
                self.cutSequenceDict["Cut Time"].append(cutTime)
                sequence = len(self.cutSequenceDict["Cut Time"])
                print(f"\nCut #{sequence} Time: {cutTime}")


                betweenCutTime = self.pressTime - self.endCutTime
                print(f"Time Between Cuts {sequence-1} and {sequence}: {betweenCutTime}")
                self.cutSequenceDict["Between Cut Time"].append(betweenCutTime)

                dirPt = [(self.startPose.x() + self.lastPose.x())/2, (self.startPose.y() + self.lastPose.y())/2]
                dir = self.rayDirection(x=dirPt[0], y=dirPt[1])[:3]
                depth, intercept = self.interception(origin=self.camera_pos, rayDirection=dir, faces=intersectFaces)
                # Need to pass in the vertices to the code
                if len(intercept) == 0:
                    print("No intercept detected")
                else:
                    # Now I need to find the min and max z but max their value slightly larger for the rectangle
                    # Depth given in world space

                    # TURN THE DRAWLINES TO TRUE SO IT DRAWS ON SCREEN
                    if self.screenType == "draw_tutorial" or self.screenType == "prune":
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
                    cubeVertices, vertices = self.determine_draw_plane(self.startPose, self.lastPose, u1, v1, minZ, u2, v2, maxZ)
                    """
                    TODO: See if I need to translate the points back to world frame for  storing before writing to a mesh file
                    """
                    print(f"Vertices: {vertices}")
                    self.cutSequenceDict["Vertices"].append(vertices)

                    self.addDrawVertices(cubeVertices)

                    # UPDATE VBO TO INCORPORATE THE NEW VERTICES
                    gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)

            # print(f"Total time for draw: {time.time() - start}\n")
            self.crossYMenu = True
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
            # 8 vertices
            # 3 pos + 3 color
            # 8 * 6 = 48

            stride = 8 * 6 * 3 # 8 vertices (cube) with 6 points (each vertex having pos (3) and color (3)) each with points a length of 3

            start = (self.drawCount - stride)  # (self.drawCount - 24) * 3
            end = self.drawCount # self.drawCount * 3

            # need to replace values at that section in draw vertices to null
            self.drawVertices[start:end] = np.zeros(end - start)
            
            self.drawCount -= stride  # because of the cube vertices count
            
            # gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)
            gl.glNamedBufferSubData(self.drawVBO, 0, self.drawVertices.nbytes, self.drawVertices)
        
        self.update() # update the GL Call

    def addLabels(self, checked=False):
        self.displayLabels = checked
        self.update()


    def setManipulationIndex(self, index=0):
        self.index = index
        self.update()


    # Takes an array of branches cutTypes for each branch
    def toPruneBranches(self, branchesToPrune):
        self.pruneBranches = branchesToPrune
        self.update()
    
    def toDrawTerms(self, termsToDraw):
        self.drawTerms = termsToDraw
        self.update()

    # Sets the wanted feature for branches to display on the screen
    def setWantedFeature(self, feature):
        self.wantedFeature = feature
        self.update() # Update the screen to draw the feature

    def setPenColor(self, decision):
        if decision == "Branch Vigor":
            self.penColor = self.vigorColor
            self.decision = "Vigor"

        elif decision == "Canopy Cut":
            self.penColor = self.canopyColor
            self.decision = "Canopy"
        else:
            self.penColor = self.spacingColor
            self.decision = "Spacing"
        
        self.cutSequenceDict["Rule"].append(self.decision)
        sequence = len(self.cutSequenceDict["Rule"])
        print(f"Cut #{sequence}: {self.decision}")
        self.cutSequenceDict["Sequence"].append(sequence) # what order of cuts are we using. 
        # self.update()s


    #############################################
    # DESCRIPTION: write the participant's cuts to a mesh file 
    #   - save mesh file of cuts as PID_#_treeName.obj in a folder under PID_#
    #############################################
    def saveCutDecisions(self):
        # Loop through dictionary containing: Cut decision (in order) and their vertices
        # Write the values to the same mesh file
        return 


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
        self.setWindowTitle("Pruning Interface")

        self.isCorrect = False
        self.index = 0
        self.correctFeature = False
        self.pageIndex = 0

        self.modules, self.labels, self.layouts, self.directories, self.trees = self.loadJSONWorkflow()

        # LIST OF MESH FILES AND SCREEN TYPES FOR THE PAGE
        self.curLabels = self.labels[self.pageIndex]
        self.screenType = self.layouts[self.pageIndex]
        
        if self.directories[self.pageIndex] == "":
            self.manipulationDir = None
        else:
            self.manipulationDir = self.directories[self.pageIndex]
        
        self.curTree = self.trees[self.pageIndex]
        
        self.screen_width = 3    # How many rows to cover
        # getting the main screen
        self.manipulation = {
            "Display": False,
            "JSON": self.jsonData["Manipulation Files"],
            "Directory": "manipulation"
        }

        self.binValues = ["SELECT", "Don't Prune", "Heading Cut", "Thinning Cut"] #
        self.binAnswers = ["SELECT", "SELECT", "SELECT"]
        self.binIndices = [0, 0, 0]
        self.branchCuts = ["Don't Prune"] * 3 # 3 branches to prune for the 6 inch rule

        self.treeMesh = None        # Load the tree and save it here
        self.skyBoxMesh = None      # save SkyBoxMesh

        # self.screenType = "scale" # Options: bin, submit, manipulation, normal 
        
        self.previousScreen = self.screenType
        self.toManipulate = True
        self.toBin = False
        self.toPrune = False
        self.toPrunBin = False
        self.submitScreen = False
        self.nextScreenManipualte = False
        self.skyBoxMesh = None # skybox is the same for every window
        self.interacted = False

        self.drawTerms = [False] * 3

        
        
        # self.retry = False

        
        # Initial meshes to load
        self.meshDictionary = self.load_mesh_files(treeFile=self.curTree, 
                                                   branchFiles=self.jsonData["Manipulation Files"], 
                                                   manipDirectory=self.manipulationDir,            
                                                   skyBoxFile="skyBox.obj")
        # self.load_mesh_files(treeFile="textureTree.obj", branchFiles=self.jsonData["Manipulation Files"], manipDirectory=self.manipulationDir, skyBoxFile="skyBox.obj")

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
        
        

    #########################################################################
    # Loading the mesh files for each value
    # INPUT:
    #   - treeFile: String of file name for the tree obj file
    #   - branchFiles: list of dictionaries containing file name and information about the branches
    #   - manipDirectory: The corresponding manipulation directory for the branchFiles
    #   - skyBoxFile: String of file name for skybox obj file
    #
    # OUTPUT:
    #   - Mesh files for tree, branches, and skybox if file is loaded in
    ##########################################################################
    def load_mesh_files(self, treeFile = None, branchFiles = None, manipDirectory = None, skyBoxFile = None):
        directory = "../obj_files/"
        # treeMesh = None
        # skyBoxMesh = None
        branchMeshes = []
        scales = []
        rotations = []
        translations = []
        featureDescription = []
        pruneDescription = []
        
        branches = {
            "Meshes": branchMeshes, 
            "Description": featureDescription,
            "Prunes": pruneDescription,
            "Scale": scales,
            "Rotation": rotations,
            "Translation": translations,
            "Answer": ""
        }

        if treeFile is not None:
            fname = directory + treeFile
            self.treeMesh = Mesh(fname)
        
        if branchFiles is not None and manipDirectory is not None:
            for branch in branchFiles[manipDirectory]["Files"]:
                fname = directory + manipDirectory + "/"+ str(branch["File Name"]) # object name
                featureDescription.append(branch["Feature Description"])
                pruneDescription.append(branch["Pruning Description"])
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
            branches["Prunes"] = pruneDescription
            branches["Answer"] = branchFiles[manipDirectory]["Answer"] # looking if the correct answer
            self.binValues = branchFiles[manipDirectory]["Bin_Values"]
        
        if skyBoxFile is not None:
            fname = directory + skyBoxFile
            self.skyBoxMesh = Mesh(fname)
        
        meshDictionary = {
            "Tree": self.treeMesh,
            "SkyBox": self.skyBoxMesh,
            "Branches": branches
        }

        return meshDictionary
    

    # LEFT HAND SIDE DEPENDENT ON WHAT TYPE OF SCREEN WE NEED TO SEE
    def leftSideScreen(self):
        self.viewGL = Test(wholeView=True, 
                           fname=self.fname, 
                           meshDictionary=self.meshDictionary,
                           jsonData=self.jsonData["Tree Files"][self.fname],
                           manipulation=self.manipulation)

        # SET THE SCREEN SIZE BASED ON IF A MANIPULATION TASK OR NOT
        if self.screenType == "manipulation":
            self.manipulationScreen()

        elif self.screenType == "rule":
            self.ruleScreen()

        elif self.screenType == "tree_terms":
            self.treeTermScreen()
        
        elif self.screenType == "bin":
            self.binScreen()
        
        elif self.screenType == "bud_features":
            self.budFeaturesScreen()

        elif self.screenType == "scale":
            self.scaleScreen()
            
        elif self.screenType == "term_cuts":
            self.termCutsScreen()

        elif self.screenType == "bin_features":
            self.binFeaturesScreen()

        else:
            self.loadTreeSectionScreen()
            # self.viewGL.setScreenProperties(screenType=self.screenType)


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
        self.viewGL.setFixedSize(350, 300) # 900, 700
        self.layout.addWidget(self.viewGL, 0, 2, 1, 1) # 1, 2, 1, 1
        self.hSlider.valueChanged.connect(self.viewGL.setTurnTableRotation) # Connect the vertical and horizontal camera sliders to the view screen
        self.viewGL.turnTableRotation.connect(self.hSlider.setValue)

        self.vSlider.valueChanged.connect(self.viewGL.setVerticalRotation)
        self.viewGL.verticalRotation.connect(self.vSlider.setValue)
        
        # Set screen properties for the whole view
        # if self.screenType == "normal":
        #     self.viewGL.setScreenProperties(screenType=self.screenType, toManipulate=False, toBin=False, toPrune=False) # default to False
        
        # elif self.screenType == "bin":
        #     self.viewGL.setScreenProperties(screenType=self.screenType, toManipulate=False, toBin=True, toPrune=False) # MIGHT TRY AND CHANGE
        
        # elif self.screenType == "manipulation":
        #     self.viewGL.setScreenProperties(screenType=self.screenType, toManipulate=True, toBin=False, toPrune=False)
            
        # elif self.screenType == "scale":
        #     self.viewGL.setScreenProperties(screenType=self.screenType, toManipulate=True, toBin=False, toPrune=True)

        # else: # submit should use previous manipulation value
        #     self.viewGL.setScreenProperties(screenType=self.screenType, toManipulate=self.toManipulate, toBin=self.toBin, toPrune=self.toPrune)
        

        # YOUR TASK BOX
        # Create a QFrame for the directory and buttons column
        self.textFrame = QFrame(self.central_widget) # self.central_widget
        self.textFrame.setFrameShape(QFrame.Shape.Box)
        self.textFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.textFrame.setFixedWidth(350) # 900
        self.layout.addWidget(self.textFrame, 1, 2, 1, 1)  # Row 1, Column 1, Span 1 row and 1 column

        # Create a QVBoxLayout for the directory and buttons column
        self.directory_layout = QVBoxLayout(self.textFrame)
        # self.progressFrame.setFixedSize(900, 500)

        # Create a QLabel to display the directory
        self.directory_label = QLabel("Your Task:")
        self.directory_label.setStyleSheet("font-size: 25px;" "font:bold")

        self.directory_layout.addWidget(self.directory_label)

        # Create a QLabel to display the task description
        # QLabel should be filled in with json data
        self.task_label = QLabel(self.labels[self.pageIndex]) # SET WITH LABELS
        self.task_label.setStyleSheet("font-size: 15px;")
        self.directory_layout.addWidget(self.task_label)
    
        self.progressFrame = QFrame(self.central_widget) # self.central_widget
        self.progressFrame.setFixedSize(350, 300) # 900, 500
        self.progressFrame.setFrameShape(QFrame.Shape.Box)
        self.progressFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.progressFrame, 2, 2, 2, 1)  # Row 1, Column 1, Span 1 row and 1 column

        self.progress_layout = QVBoxLayout(self.progressFrame)
        self.progress_label = QLabel("Your Progress")
        self.progress_label.setStyleSheet("font-size: 25px;" "font:bold")
        self.progress_layout.addWidget(self.progress_label)

        text = f"Module: {self.modules[self.pageIndex]}" 
        self.moduleLabel = QLabel(text)
        self.moduleLabel.setStyleSheet("font-size: 15px;")
        self.progress_layout.addWidget(self.moduleLabel)

        self.progressBar = QProgressBar(self.central_widget)
        self.progressBar.setRange(0, len(self.modules) - 1)
        # self.progressBar.setGeometry(0, 0, 50, 30)
        self.progressBar.setFixedSize(300, 50) # 800, 50
        self.progressBar.setValue(self.pageIndex)
        # TARDIS BLUE 003B6F
        # Chunk width = 120px 
        # "QProgressBar::chunk {background-color: #4F9153; width: 120px; margin: 0.5px;}"
        self.progressBar.setStyleSheet("QProgressBar {width: 30px; text-align: center;}\n"
                                       "QProgressBar::chunk {background-color: #4F9153; margin: 0.5px;}") 
        #{border: 2px solid #2196F3; border-radius: 5px; background-color: #E0E0E0;}
        # background-color: #4F9153; 
        self.progress_layout.addWidget(self.progressBar)

        self.nextButton = QPushButton("Next") # Make a blank button
        self.nextButton.setStyleSheet("QPushButton {font-size: 25px;" "font:bold}\n"
                                      "QPushButton::pressed {background-color: #4F9153;}") # 
        self.nextButton.clicked.connect(self.nextPageButtonClicked)
        if self.screenType == "end_section":
            self.nextButton.setEnabled(True)
        else:
            self.nextButton.setEnabled(False)
        self.nextButton.setFixedSize(150, 50) # 300, 100
        self.nextButton.setCheckable(True)

        self.nextLabel = QLabel() #
        
        self.progress_layout.addWidget(self.nextLabel)
        self.progress_layout.addWidget(self.nextButton)
        

    def createSlider(self, camera=True, horizontal=True, endRange=0):
        if horizontal:
            slider = QSlider(Qt.Horizontal)
        else:
            slider = QSlider(Qt.Vertical)
            slider.setStyleSheet("QSlider::handle:vertical {background-color: blue; border: 1px solid; height: 15px; width: 25px; margin: -15px 0px;}")
            # "QSlider::groove:vertical {background: darkgray; border: 1px; height: 10px; margin: 2px 0;}"
            # "QSlider::groove:vertical {border: 1px; height: 30px; margin: 0px}\n"
        # slider.setRange(0, 360) # 0 - 360*16
        if camera:
            if horizontal:
                slider.setRange(-30, 30)
            else:
                slider.setRange(-15, 15)
            slider.setSingleStep(1) # 
            # slider.setPageStep(10)
            slider.setPageStep(5)
            slider.setTickPosition(QSlider.TicksBothSides) # TicksBelow
            if horizontal:
                slider.setStyleSheet("QSlider::handle:horizontal {background-color: blue; border: 1px solid; height: 15px; width: 25px; margin: -15px 0px;}")
                # "QSlider::groove:horizontal {border: 1px; height: 30px; margin: 0px}\n"
        else:
            if endRange == 0:
                slider.setRange(0, 10)
            else:
                slider.setRange(0, endRange) # END RANGE IS INCLUSIVE
            slider.setSingleStep(1) # 
            # slider.setPageStep(10)
            slider.setPageStep(1)
            slider.setTickPosition(QSlider.TicksBelow) 
            slider.setStyleSheet("QSlider::handle:horizontal {background-color: black; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")
            
            # slider.setStyleSheet("QSlider::groove:horizontal {border: 1px; height: 30px; margin: 0px}\n"
            #                  "QSlider::handle:horizontal {background-color: black; border: 1px solid; height: 30px; width: 30px; margin: -15px 0px;}")

        # slider.setGeometry(50, 50, 300, 300)
        # slider.setStyleSheet("QSlider::handle:horizontal { min-height: 80px; max-height: 80px;}")
        
        # "QSlider::groove:horizontal {height: 10px; margin: 0 0;}\n"
        # "QSlider::handle:horizontal {background-color: black; border: 1px; height: 
        # 15px; width: 15px; margin: 0 0;}\n"
        # ""
        return slider

    def setScaleLabels(self, value):
        self.index = value
        # # Check if user has moved anything here
        if not self.interacted:
            self.interacted = True
            self.nextButton.setEnabled(True)

    
    def hCameraSliderClicked(self):
        self.hSlider.setStyleSheet("QSlider::handle:horizontal {background-color: green; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")

    def hCameraSliderReleased(self):
        self.interacted = True
        if self.screenType == "normal":
            self.isCorrect = True
            self.nextButton.setEnabled(True)
        self.hSlider.setStyleSheet("QSlider::handle:horizontal {background-color: blue; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")


    def vCameraSliderClicked(self):
        self.vSlider.setStyleSheet("QSlider::handle:vertical {background-color: green; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")
 
    def vCameraSliderReleased(self):
        self.interacted = True
        if self.screenType == "normal":
            self.isCorrect = True
            self.nextButton.setEnabled(True)
        self.vSlider.setStyleSheet("QSlider::handle:vertical {background-color: blue; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")

    def branchScaleSliderClicked(self):
        self.interacted = True
        self.scaleSlider.setStyleSheet("QSlider::handle:horizontal {background-color: green; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")

    def branchScaleSliderReleased(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        self.scaleSlider.setStyleSheet("QSlider::handle:horizontal {background-color: black; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")

    def branchManipSliderClicked(self):
        self.interacted = True
        self.manipulationSlider.setStyleSheet("QSlider::handle:horizontal {background-color: green; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")

    def branchManipSliderReleased(self):
        self.interacted = True
        self.manipulationSlider.setStyleSheet("QSlider::handle:horizontal {background-color: black; border: 1px solid; height: 25px; width: 30px; margin: -10px 0px;}")

    
    def setManipulationSlider(self, value):
        self.index = value
        # self.nextButton.setEnabled(True)
        self.manipulationSlider.setValue(value) # set the slider value
    
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
            # self.isCorrect = True
            self.screen_width = 3
            self.glWidgetTree.setScreenProperties(screenType=self.screenType) # default to False
            self.toManipulate = False
            self.toBin = False
            self.toPrune = False

        elif self.screenType == "draw_tutorial":
            self.screen_width = 3
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, drawLines=True)
            self.drawLines = True
        
        elif self.screenType == "tree_terms":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, termDraw=True)
            self.termDraw = True
        
        elif self.screenType == "end_section":
            self.isCorrect = True
            self.interacted = True
            self.screen_width = 3
            self.glWidgetTree.setScreenProperties(screenType="normal") # same as normal

        elif self.screenType == "bin":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toBin=True) # MIGHT TRY AND CHANGE
            self.toManipulate = False
            self.toBin = True
            self.toPrune = False

        elif self.screenType == "bin_features" or self.screenType == "bud_features":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toBinFeatures=True) # MIGHT TRY AND CHANGE
            self.toBinFeatures = True
            self.toPrune = False
            self.toManipulate = False
        
        elif self.screenType == "manipulation":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toManipulate=True)
            self.toManipulate = True
            self.toBin = False 
            self.toPrune = False

        elif self.screenType == "scale":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toManipulate=True, toPrune=True)
            self.toManipulate = True
            self.toBin = False 
            self.toPrune = True

        elif self.screenType == "rule":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toBin=True, toPruneBin=True)
            self.toManipulate = False
            self.toBin = True 
            self.toPrune = False # TO CHANGE
            self.toPruneBin = True
        
        elif self.screenType == "term_cuts":
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toManipulate=True, toPrune=True)
            self.toManipulate = True
            self.toBin = False 
            self.toPrune = True # TO CHANGE
            self.toPruneBin = True

        else: # submit should use previous manipulation value
            self.screen_width = 2
            self.glWidgetTree.setScreenProperties(screenType=self.screenType, toManipulate=self.toManipulate, toBin=self.toBin, toPrune=self.toPrune)
        
        # self.glWidget.setFixedSize(2820, 1850) # can I find the aspect
        # self.layout.addWidget(self.glWidget)
        self.layout.addWidget(self.glWidgetTree, 0, 1, self.screen_width, 1) # r=0, c=1, rs = 3, cs = 1
        
        # LABEL BUTTON
        self.labelButton = QPushButton("Labels On") # Make a blank button
        self.labelButton.setStyleSheet("font-size: 25px;" "font:bold")
        self.labelButton.setCheckable(True)
        self.labelButton.setFixedSize(150, 50) # 300, 100
        self.labelButton.clicked.connect(self.labelButtonClicked)
        self.hLayout.addWidget(self.labelButton)

        # SUBMIT BUTTON
        # if self.screenType == "normal": # should only show when both are false
        #     self.submitButton = QPushButton("Next") # Make a blank button
        #     self.submitButton.setStyleSheet("font-size: 25px;" "font:bold", "QPushButton::pressed {background-color: #4F9153;}")
        #     self.submitButton.clicked.connect(self.nextPageButtonClicked) # TO CONNECT THE BUTTON WITH SCREEN
        #     self.submitButton.setFixedSize(300, 100)
        #     self.hLayout.addWidget(self.submitButton)

            # UNDO BUTTON
        if self.screenType == "draw_tutorial" or self.screenType == "prune":
            self.undoButton = QPushButton("Undo")
            self.undoButton.setStyleSheet("font-size: 25px;" "font:bold")
            self.undoButton.setFixedSize(150, 50) # 300, 100
            self.undoButton.clicked.connect(self.glWidgetTree.undoDraw)
            self.undoButton.clicked.connect(self.undoClicked)
            self.hLayout.addWidget(self.undoButton)

            # font = QFont()
            # font.setPointSize(font.pointSize()+2)
            # self.answerText = QLabel("")

            # # create a dropdown menu with different color
            # self.pruningRules = ["Vigor", "Canopy", "Spacing"]

            # self.penLabel = QLabel("Pruning Cut:")
            # self.penLabel.setStyleSheet("font-size: 25px;" "font:bold;" "color: white")
            # self.hLayout.addWidget(self.penLabel)

            # self.vigorRadio = QRadioButton("Branch Vigor")
            # self.vigorRadio.setChecked(True)
            # self.vigorRadio.setStyleSheet("font-size: 20px;" "color: #ff5f1f")
            # self.vigorRadio.toggled.connect(lambda: self.radioButtonClicked("Branch Vigor"))
            
            # self.canopyRadio = QRadioButton("Canopy Cut")
            # self.canopyRadio.setStyleSheet("font-size: 20px;" "color: #00f0ff")
            # self.canopyRadio.toggled.connect(lambda: self.radioButtonClicked("Canopy Cut"))
            
            # self.spacingRadio = QRadioButton("Bud Spacing")
            # self.spacingRadio.setStyleSheet("font-size: 20px;" "color: #21fc0d")
            # self.spacingRadio.toggled.connect(lambda: self.radioButtonClicked("Bud Spacing"))

            # self.hLayout.addWidget(self.vigorRadio)
            # self.hLayout.addWidget(self.canopyRadio)
            # self.hLayout.addWidget(self.spacingRadio)

            # self.radioButtonClicked("Branch Vigor")

            

            # self.dropDownPen = QComboBox()
            # self.dropDownPen.setFixedSize(300, 50)
            # self.dropDownPen.setFont(font)
            # self.dropDownPen.addItems()
            # self.dropDownPen.setCurrentIndex(self.binIndices[0])
            # self.dropDownPen.activated.connect(self.dropDownTextSelected)
            # self.binLayout.addWidget(self.dropDown, 2, 0, Qt.AlignTop | Qt.AlignCenter)
            

        
        # self.layout.addWidget(self.labelButton, 0, 1, 1, 1)
        self.layout.addLayout(self.hLayout, 0, 1, Qt.AlignTop | Qt.AlignLeft)

        # VERTICAL SLIDER
        self.vSlider = self.createSlider(camera=True, horizontal=False)
        self.vSlider.valueChanged.connect(self.glWidgetTree.setVerticalRotation)
        self.vSlider.sliderPressed.connect(self.vCameraSliderClicked)
        self.vSlider.sliderReleased.connect(self.vCameraSliderReleased)
        self.glWidgetTree.verticalRotation.connect(self.vSlider.setValue)
        # self.layout.addWidget(self.vSlider)
        self.layout.addWidget(self.vSlider, 0, 0, self.screen_width, 1) # 0, 0, 3, 1
        
        # HORIZONTAL SLIDER
        self.hSlider = self.createSlider(camera=True, horizontal=True)
        self.hSlider.valueChanged.connect(self.glWidgetTree.setTurnTableRotation)
        self.hSlider.sliderPressed.connect(self.hCameraSliderClicked)
        self.hSlider.sliderReleased.connect(self.hCameraSliderReleased)
        self.glWidgetTree.turnTableRotation.connect(self.hSlider.setValue)
        self.layout.addWidget(self.hSlider, self.screen_width, 1, 1, 1) # 3 1 1 1


    def undoClicked(self):
        self.interacted = True
        if self.screenType == "draw_tutorial":
            self.isCorrect = True
            self.nextButton.setEnabled(True)


    def radioButtonClicked(self, option):
        if option:
            self.glWidgetTree.setPenColor(option)



    
    def manipulationScreen(self):
        
        self.loadTreeSectionScreen() # LOAD THE TREE SECTION

        self.manipulationFrame = QFrame(self.central_widget) # self.central_widget
        self.manipulationFrame.setFrameShape(QFrame.Shape.Box)
        self.manipulationFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.manipulationFrame.setFixedHeight(300) # 500
        self.layout.addWidget(self.manipulationFrame, self.screen_width+1, 1, 1, 1) # span the screen width but start below the sliders
        self.manipLayout = QGridLayout(self.manipulationFrame)


        self.answerText = QLabel("")
    
        # TO CHANGE THE NAME OF THE SLIDER BASED ON WHAT WE ARE MANIPULATING
        self.manipulationLabel = QLabel("Branch Manipulation Slider:") 
        self.manipulationLabel.setStyleSheet("font-size: 25px;" "font:bold")
        self.manipLayout.addWidget(self.manipulationLabel, 0, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        # MANIPULATION SLIDER
        sliderLength = len(self.meshDictionary["Branches"]["Description"]) - 1 # END RANGE IS INCLUSIVE
        self.manipulationSlider = self.createSlider(camera=False, horizontal=True, endRange=sliderLength)
        # Connect the slider to values on the screen
        self.manipulationSlider.valueChanged.connect(self.glWidgetTree.setManipulationIndex)
        self.manipulationSlider.valueChanged.connect(self.viewGL.setManipulationIndex)
        self.manipulationSlider.sliderPressed.connect(self.branchManipSliderClicked)
        self.manipulationSlider.sliderReleased.connect(self.branchManipSliderReleased)
        
        self.glWidgetTree.manipulationIndex.connect(self.manipulationSlider.setValue)
        self.viewGL.manipulationIndex.connect(self.manipulationSlider.setValue)
        self.manipulationSlider.valueChanged.connect(self.scaleIndex)
        self.manipulationSlider.setValue(self.index)    # Set the slider to a particular value (for reloading)
        self.manipLayout.addWidget(self.manipulationSlider, 0, 1, 1, 2) # 3 1 1 1

        # self.manipulationLabel.setBuddy(self.manipulationSlider)

        # placeholder = QLabel()
        # self.manipLayout.addWidget(placeholder, 1, 1, 1, 3)

        # SUBMIT BUTTON
        self.submitButton = QPushButton("Submit") # Make a blank button
        self.submitButton.setStyleSheet("QPushButton {font-size: 25px;" "font:bold}\n"
                                        "QPushButton::pressed {background-color: #4F9153;}") # 
        
        self.submitButton.setFixedSize(150, 50) # 300, 100
        # self.submitButton.clicked.connect(self.submitButtonClicked)
        self.submitButton.clicked.connect(self.submit)
        self.manipLayout.addWidget(self.submitButton, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        
        self.answerText.setStyleSheet("font-size: 25px;")
        self.manipLayout.addWidget(self.answerText, 1, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)
        # self.manipLayout.addWidget(self.submitButton, 2, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)
        

    def checkAnswer(self):
        text = ""
        if self.screenType == "manipulation": # self.glWidgetTree.toManipulate
            if self.correctFeature:
                # print("IS THE CORRECT FEATURE")
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Correct"]
                self.isCorrect = True
                self.nextButton.setEnabled(True)
                # self.submitButton = QPushButton("Next") 
                # self.submitButton.clicked.connect(self.nextPageButtonClicked)
            else:
                # print("NOT THE CORRECT FEATURE")
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Incorrect"]
                self.nextButton.setEnabled(False)
                
        elif self.screenType == "bin": # CHECK THE BIN ANSWERS
            correct = self.jsonData["Manipulation Files"][self.manipulationDir]["Answer"]
            if self.compareBinAnswers(correct, self.binAnswers):
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Correct"]
                self.isCorrect = True
                self.nextButton.setEnabled(True)
                # self.submitButton = QPushButton("Next") 
                # self.submitButton.clicked.connect(self.nextPageButtonClicked)
            else:
                text = self.jsonData["Manipulation Files"][self.manipulationDir]["Incorrect"]
                self.nextButton.setEnabled(False)
        return text


    def scaleIndex(self, value):
        self.index = value
        self.answerText.setText("")
        self.answerText.setStyleSheet("font-size: 25px;")


    def submit(self):
        # Grab the value to check if correct
        self.correctFeature = self.glWidgetTree.correctFeature
        text = self.checkAnswer()
        self.answerText.setText(text)
        self.answerText.setStyleSheet("font-size: 25px;")
        

    def pruneButtonClicked(self):

        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        if self.branch1PruneButton.isChecked():
            self.branch1PruneButton.setText("Don't Prune")
            self.branch1PruneButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                                  "QPushButton::pressed {background-color: #4F9153;}")
            self.branchCuts[0] = "Heading Cut" 
            # TO DO: Call function in GLWidget to prune the branch
        else:
            self.branch1PruneButton.setText("Prune")
            self.branchCuts[0] = "Don't Prune"
        
        self.glWidgetTree.toPruneBranches(self.branchCuts)


    def pruneButtonClicked2(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        if self.branch2PruneButton.isChecked():
            self.branch2PruneButton.setText("Don't Prune")
            self.branch2PruneButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                                  "QPushButton::pressed {background-color: #4F9153;}")
            self.branchCuts[1] = "Heading Cut"
            # TO DO: Call function in GLWidget to prune the branch
        else:
            self.branch2PruneButton.setText("Prune")
            self.branchCuts[1] = "Don't Prune"
        
        self.glWidgetTree.toPruneBranches(self.branchCuts)

    
    def pruneButtonClicked3(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        if self.branch3PruneButton.isChecked():
            self.branch3PruneButton.setText("Don't Prune")
            self.branch3PruneButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                                  "QPushButton::pressed {background-color: #4F9153;}")
            self.branchCuts[2] = "Heading Cut" 
            # TO DO: Call function in GLWidget to prune the branch

        else:
            self.branch3PruneButton.setText("Prune")
            self.branchCuts[2] = "Don't Prune"
        
        self.glWidgetTree.toPruneBranches(self.branchCuts)



    def ruleScreen(self):
        self.loadTreeSectionScreen()
        
        # Set the screen frame
        self.ruleFrame = QFrame(self.central_widget) # self.central_widget
        self.ruleFrame.setFrameShape(QFrame.Shape.Box)
        self.ruleFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.ruleFrame.setFixedHeight(300) # 500 
        self.layout.addWidget(self.ruleFrame, self.screen_width+1, 1, 1, 1) # Where on the screen we add
        self.ruleLayout = QGridLayout(self.ruleFrame)

        self.inchLabel = QLabel("6 Inch Rule: Prune back branches to within 6 inches of the secondary branch")
        self.inchLabel.setStyleSheet("font-size: 25px;" "font:bold")
        self.ruleLayout.addWidget(self.inchLabel, 0, 0, 1, 3, Qt.AlignBottom | Qt.AlignCenter)

        # make the 3 labels and buttons for the individual branches that should be pruned when clicked
        self.branch1Label = QLabel("Branch 1")
        self.branch1Label.setStyleSheet("font-size: 20px;" "font:bold")
        self.ruleLayout.addWidget(self.branch1Label, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        
        self.branch1PruneButton = QPushButton("Prune")
        self.branch1PruneButton.setCheckable(True)
        self.branch1PruneButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                              "QPushButton::pressed {background-color: #4F9153;}")
        self.branch1PruneButton.setFixedSize(150, 50) # 300, 100
        self.branch1PruneButton.clicked.connect(self.pruneButtonClicked)
        self.ruleLayout.addWidget(self.branch1PruneButton, 2, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        # BRANCH 2
        self.branch2Label = QLabel("Branch 2")
        self.branch2Label.setStyleSheet("font-size: 20px;" "font:bold")
        self.ruleLayout.addWidget(self.branch2Label, 1, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        self.branch2PruneButton = QPushButton("Prune") 
        self.branch2PruneButton.setCheckable(True)
        self.branch2PruneButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                              "QPushButton::pressed {background-color: #4F9153;}")
        self.branch2PruneButton.setFixedSize(150, 50) # 300, 100
        self.branch2PruneButton.clicked.connect(self.pruneButtonClicked2)
        self.ruleLayout.addWidget(self.branch2PruneButton, 2, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        # BRANCH 3
        self.branch3Label = QLabel("Branch 3")
        self.branch3Label.setStyleSheet("font-size: 20px;" "font:bold")
        self.ruleLayout.addWidget(self.branch3Label, 1, 2, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        self.branch3PruneButton = QPushButton("Prune") 
        self.branch3PruneButton.setCheckable(True)
        self.branch3PruneButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                              "QPushButton::pressed {background-color: #4F9153;}")
        self.branch3PruneButton.setFixedSize(150, 50) # 300, 100
        self.branch3PruneButton.clicked.connect(self.pruneButtonClicked3)
        self.ruleLayout.addWidget(self.branch3PruneButton, 2, 2, 1, 1, Qt.AlignBottom | Qt.AlignCenter)



    """
        TREE TERMINOLOGY SCREEN AND HELPER FUNCTIONS
    """
    def treeTermScreen(self):
        self.loadTreeSectionScreen()
        
        # Set the screen frame
        self.treeTermFrame = QFrame(self.central_widget) # self.central_widget
        self.treeTermFrame.setFrameShape(QFrame.Shape.Box)
        self.treeTermFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.treeTermFrame.setFixedHeight(300) # 500
        self.layout.addWidget(self.treeTermFrame, self.screen_width+1, 1, 1, 1) # Where on the screen we add
        self.treeTermLayout = QGridLayout(self.treeTermFrame)

        self.termLabel = QLabel("Tree Terminology")
        self.termLabel.setStyleSheet("font-size: 25px;" "font:bold")
        self.treeTermLayout.addWidget(self.termLabel, 0, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        self.descriptionLabel = QLabel("")
        self.descriptionLabel.setStyleSheet("font-size: 25px;")
        self.treeTermLayout.addWidget(self.descriptionLabel, 0, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)

        # make the 3 labels and buttons for the individual branches that should be pruned when clicked
        self.trunkLabel = QLabel("Trunk")
        self.trunkLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.treeTermLayout.addWidget(self.trunkLabel, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        
        self.trunkButton = QPushButton("Show")
        self.trunkButton.setCheckable(True)
        self.trunkButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                              "QPushButton::pressed {background-color: #4F9153;}")
        self.trunkButton.setFixedSize(150, 50) # 150, 50
        self.trunkButton.clicked.connect(self.trunkButtonClicked)
        self.treeTermLayout.addWidget(self.trunkButton, 2, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        # BRANCH 2
        self.secondaryLabel = QLabel("Secondary Branch")
        self.secondaryLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.treeTermLayout.addWidget(self.secondaryLabel, 1, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        self.secondaryButton = QPushButton("Show") 
        self.secondaryButton.setCheckable(True)
        self.secondaryButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                              "QPushButton::pressed {background-color: #4F9153;}")
        self.secondaryButton.setFixedSize(150, 50) # 300, 100
        self.secondaryButton.clicked.connect(self.secondaryButtonClicked)
        self.treeTermLayout.addWidget(self.secondaryButton, 2, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        # BRANCH 3
        self.tertiaryLabel = QLabel("Tertiary Branch")
        self.tertiaryLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.treeTermLayout.addWidget(self.tertiaryLabel, 1, 2, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        self.tertiaryButton = QPushButton("Show") 
        self.tertiaryButton.setCheckable(True)
        self.tertiaryButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                              "QPushButton::pressed {background-color: #4F9153;}")
        self.tertiaryButton.setFixedSize(150, 50)
        self.tertiaryButton.clicked.connect(self.tertiaryButtonClicked)
        self.treeTermLayout.addWidget(self.tertiaryButton, 2, 2, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


    def trunkButtonClicked(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        if self.trunkButton.isChecked():
            self.descriptionLabel.setText("Trunk, primary branch, or leader is the main structure that connects to the roots")
            self.descriptionLabel.setStyleSheet("font-size: 20px;")

            self.trunkButton.setText("Hide")
            self.trunkButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                            "QPushButton::pressed {background-color: #4F9153;}")
            self.drawTerms[0] = True
            # Don't highlight anything else until the branches are clicked
            # self.trunkButton.setEnabled(False)
            self.secondaryButton.setEnabled(False)
            self.tertiaryButton.setEnabled(False)
            # TO DO: Call function in GLWidget to prune the branch

        else:
            self.descriptionLabel.setText("")
            self.trunkButton.setText("Show")
            self.drawTerms[0] = False

            # self.trunkButton.setEnabled(True)
            self.secondaryButton.setEnabled(True)
            self.tertiaryButton.setEnabled(True)
        
        self.glWidgetTree.toDrawTerms(self.drawTerms)
    

    def secondaryButtonClicked(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        if self.secondaryButton.isChecked():
            self.descriptionLabel.setText("Secondary Branches are wired down support branches growing out from the trunk")
            self.descriptionLabel.setStyleSheet("font-size: 20px;")
            self.secondaryButton.setText("Hide")
            self.secondaryButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                            "QPushButton::pressed {background-color: #4F9153;}")
            self.drawTerms[1] = True
            # Don't highlight anything else until the branches are clicked
            self.trunkButton.setEnabled(False)
            # self.secondaryButton.setEnabled(False)
            self.tertiaryButton.setEnabled(False)
            # TO DO: Call function in GLWidget to prune the branch

        else:
            self.descriptionLabel.setText("")
            self.secondaryButton.setText("Show")
            self.drawTerms[1] = False

            self.trunkButton.setEnabled(True)
            # self.secondaryButton.setEnabled(True)
            self.tertiaryButton.setEnabled(True)
        
        self.glWidgetTree.toDrawTerms(self.drawTerms)
  


    def tertiaryButtonClicked(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        if self.tertiaryButton.isChecked():
            self.descriptionLabel.setText("Tertiary branches grow from secondary branches and produce fruit and leaves.")
            self.descriptionLabel.setStyleSheet("font-size: 20px;")
            self.tertiaryButton.setText("Hide")
            self.tertiaryButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                            "QPushButton::pressed {background-color: #4F9153;}")
            self.drawTerms[2] = True
            # Don't highlight anything else until the branches are clicked
            self.trunkButton.setEnabled(False)
            self.secondaryButton.setEnabled(False)
            # self.tertiaryButton.setEnabled(False)
            # TO DO: Call function in GLWidget to prune the branch

        else:
            self.descriptionLabel.setText("")
            self.tertiaryButton.setText("Show")
            self.drawTerms[2] = False
            self.trunkButton.setEnabled(True)
            self.secondaryButton.setEnabled(True)
            # self.tertiaryButton.setEnabled(True)
        
        self.glWidgetTree.toDrawTerms(self.drawTerms)





    
    """
        TERM CUTS SCREEN AND HELPER FUNCTIONS:
            - termCutsScreen
            - headingButtonClicked
            - thinningButtonClicked
    """ 
    def termCutsScreen(self):
        self.loadTreeSectionScreen()

        self.cutFrame = QFrame(self.central_widget)
        self.cutFrame.setFrameShape(QFrame.Shape.Box)
        self.cutFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.cutFrame.setFixedHeight(300) # 300
        self.layout.addWidget(self.cutFrame, self.screen_width+1, 1, 1, 1)
        self.cutLayout = QGridLayout(self.cutFrame)

        # Label explaining the type of cuts
        self.descriptionLabel = QLabel("Pruning Cut Description:")
        self.descriptionLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.cutLayout.addWidget(self.descriptionLabel, 0, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.cutLabel = QLabel("")
        self.cutLabel.setStyleSheet("font-size: 20px;")
        self.cutLayout.addWidget(self.cutLabel, 0, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)

        
        self.headingButton = QPushButton("Heading Cut") 
        self.headingButton.setCheckable(False)
        self.headingButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.headingButton.setFixedSize(150, 50)
        self.headingButton.clicked.connect(self.headingButtonClicked)
        self.cutLayout.addWidget(self.headingButton, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.thinningButton = QPushButton("Thinning Cut") 
        self.thinningButton.setCheckable(False)
        self.thinningButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.thinningButton.setFixedSize(150, 50)
        self.thinningButton.clicked.connect(self.thinningButtonClicked)
        self.cutLayout.addWidget(self.thinningButton, 1, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)



    def headingButtonClicked(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        self.cutLabel.setText("A Heading Cut leaves part of the branch but removes the end")
        self.cutLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setManipulationIndex(index=1)
        

    def thinningButtonClicked(self):
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
        self.cutLabel.setText("A Thinning Cut removes the branch completely from the secondary branch")
        self.cutLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setManipulationIndex(index=2)



    """
        BIN FEATURES SCREEN AND HELPER FUNCTIONS:
            - binFeaturesScreen
            - weakButtonClicked
            - strongButtonClicked
            - vigorousButtonClicked
    """      

    def binFeaturesScreen(self):
        self.loadTreeSectionScreen()

        self.binFeatureFrame = QFrame(self.central_widget)
        self.binFeatureFrame.setFrameShape(QFrame.Shape.Box)
        self.binFeatureFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.binFeatureFrame.setFixedHeight(300) # 500
        self.layout.addWidget(self.binFeatureFrame, self.screen_width+1, 1, 1, 1)
        self.binFeatureLayout = QGridLayout(self.binFeatureFrame)

        # Label explaining the type of cuts
        self.vigorLabel = QLabel("Branch Vigor, or a branch's health, is a continuous scale with three levels:")
        self.vigorLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.binFeatureLayout.addWidget(self.vigorLabel, 0, 0, 1, 3, Qt.AlignBottom | Qt.AlignCenter)


        self.featureLabel = QLabel("Branch Vigor Description:")
        self.featureLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.binFeatureLayout.addWidget(self.featureLabel, 2, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.descriptionLabel = QLabel("")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.binFeatureLayout.addWidget(self.descriptionLabel, 2, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)

      
        self.weakButton = QPushButton("Weak") 
        self.weakButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.weakButton.setFixedSize(150, 50)
        self.weakButton.clicked.connect(self.weakButtonClicked)
        self.binFeatureLayout.addWidget(self.weakButton, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.strongButton = QPushButton("Strong") 
        self.strongButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.strongButton.setFixedSize(150, 50)
        self.strongButton.clicked.connect(self.strongButtonClicked)
        self.binFeatureLayout.addWidget(self.strongButton, 1, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.vigorousButton = QPushButton("Vigorous") 
        self.vigorousButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.vigorousButton.setFixedSize(150, 50)
        self.vigorousButton.clicked.connect(self.vigorousButtonClicked)
        self.binFeatureLayout.addWidget(self.vigorousButton, 1, 2, 1, 1, Qt.AlignBottom | Qt.AlignCenter)



    def weakButtonClicked(self):
        self.descriptionLabel.setText("Weak branches are shorter, thinner, and have few buds")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setWantedFeature("Weak")
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)

    def strongButtonClicked(self):
        self.descriptionLabel.setText("Strong branches are thicker, longer, with several buds")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setWantedFeature("Strong")
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)

    def vigorousButtonClicked(self):
        self.descriptionLabel.setText("Vigorous branches are incredibly long, thick, with few buds and large bud spacing")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setWantedFeature("Vigorous")
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)


    """BUD FEATURES SCREEN"""
    def budFeaturesScreen(self):
        self.loadTreeSectionScreen()

        self.binFeatureFrame = QFrame(self.central_widget)
        self.binFeatureFrame.setFrameShape(QFrame.Shape.Box)
        self.binFeatureFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.binFeatureFrame.setFixedHeight(300) # 500
        self.layout.addWidget(self.binFeatureFrame, self.screen_width+1, 1, 1, 1)
        self.binFeatureLayout = QGridLayout(self.binFeatureFrame)

        # Label explaining the type of cuts
        self.spacingLabel = QLabel("Buds need enough space (~2-3 inches) to produce apples:")
        self.spacingLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.binFeatureLayout.addWidget(self.spacingLabel, 0, 0, 1, 3, Qt.AlignBottom | Qt.AlignCenter)


        self.featureLabel = QLabel("Bud Spacing:")
        self.featureLabel.setStyleSheet("font-size: 20px;" "font:bold")
        self.binFeatureLayout.addWidget(self.featureLabel, 2, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.descriptionLabel = QLabel("")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.binFeatureLayout.addWidget(self.descriptionLabel, 2, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)

      
        self.spaceButton = QPushButton("Enough Space") 
        self.spaceButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.spaceButton.setFixedSize(150, 50)
        self.spaceButton.clicked.connect(self.spaceButtonClicked)
        self.binFeatureLayout.addWidget(self.spaceButton, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


        self.closeButton = QPushButton("Too Close") 
        self.closeButton.setStyleSheet("QPushButton {font-size: 15px;" "font:bold}\n"
                                         "QPushButton::pressed {background-color: #4F9153;}")
        self.closeButton.setFixedSize(150, 50)
        self.closeButton.clicked.connect(self.closeButtonClicked)
        self.binFeatureLayout.addWidget(self.closeButton, 1, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)


    def closeButtonClicked(self):
        self.descriptionLabel.setText("Buds are less than 2-3 inches apart")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setWantedFeature("Too Close")
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)
    
    def spaceButtonClicked(self):
        self.descriptionLabel.setText("Buds are have more than 2-3 inches apart")
        self.descriptionLabel.setStyleSheet("font-size: 20px;")
        self.glWidgetTree.setWantedFeature("Enough Space")
        self.interacted = True
        self.isCorrect = True
        self.nextButton.setEnabled(True)


    """
        SCALE SCREEN FOR MANIPULATION TASKS
    """    
    def scaleScreen(self):
        self.loadTreeSectionScreen()

        # Set the screen frame
        self.scaleFrame = QFrame(self.central_widget) # self.central_widget
        self.scaleFrame.setFrameShape(QFrame.Shape.Box)
        self.scaleFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.scaleFrame.setFixedHeight(300) # 500
        self.layout.addWidget(self.scaleFrame, self.screen_width+1, 1, 1, 1) # Where on the screen we add
        self.scaleLayout = QGridLayout(self.scaleFrame)

        # add the slider to the screen
        # make to however many branches I have
        # END RANGE IS INCLUSIVE
        
        # text = "Branch Vigor Level: " + self.meshDictionary["Branches"]["Description"][0]
        # self.scaleLabel = QLabel(text)
        # self.scaleLabel.setStyleSheet("font-size: 20px;" "font:bold")
        # self.scaleLayout.addWidget(self.scaleLabel, 1, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter) # where in the small table we add        
        
        # # Maybe I should make a button appear if they move the slider
        # pruneText = "Pruning Decision: " + self.meshDictionary["Branches"]["Prunes"][0]
        # self.pruneScaleLabel = QLabel(pruneText)
        # self.pruneScaleLabel.setStyleSheet("font-size: 20px;" "font:bold")
        # self.scaleLayout.addWidget(self.pruneScaleLabel, 1, 2, 1, 1, Qt.AlignBottom | Qt.AlignCenter) # where in the small table we add        

        self.scaleSliderLabel = QLabel("Branch Manipulation Slider:")
        self.scaleSliderLabel.setStyleSheet("font-size: 25px;" "font:bold")
        self.scaleLayout.addWidget(self.scaleSliderLabel, 0, 0, 1, 1, Qt.AlignBottom | Qt.AlignCenter)

        sliderLength = len(self.meshDictionary["Branches"]["Description"]) - 1 
        self.scaleSlider = self.createSlider(camera=False, horizontal=True, endRange=sliderLength)
        # Connect slider to value
        self.scaleSlider.valueChanged.connect(self.glWidgetTree.setManipulationIndex)
        self.scaleSlider.valueChanged.connect(self.viewGL.setManipulationIndex)
        self.scaleSlider.sliderPressed.connect(self.branchScaleSliderClicked)
        self.scaleSlider.sliderReleased.connect(self.branchScaleSliderReleased)
        self.glWidgetTree.manipulationIndex.connect(self.scaleSlider.setValue)
        self.viewGL.manipulationIndex.connect(self.scaleSlider.setValue)
        self.scaleSlider.valueChanged.connect(self.setScaleLabels) # get the index for the labels
        
        self.scaleLayout.addWidget(self.scaleSlider, 0, 1, 1, 2) # 3 1 1 1
        # self.scaleSliderLabel.setBuddy(self.scaleSlider)
        self.scaleLayout.addWidget(self.scaleSlider, 0, 1)

        empty = QLabel("") 
        self.scaleLayout.addWidget(empty, 1, 1, 1, 1, Qt.AlignBottom | Qt.AlignCenter)



    """
        BIN SCREEN AND HELPER FUNCTIONS:
            - binScreen
            - dropDownTextSelected
            - setDropDownValues
    """
    def binScreen(self):
        self.loadTreeSectionScreen()

        # self.binFrame
        self.binFrame = QFrame(self.central_widget) # self.central_widget
        self.binFrame.setFrameShape(QFrame.Shape.Box)
        self.binFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.binFrame.setFixedHeight(300)
        self.layout.addWidget(self.binFrame, self.screen_width+1, 1, 1, 1) # Where on the screen we add
        self.binLayout = QGridLayout(self.binFrame)
        
        # Setting the font
        font = QFont()
        font.setPointSize(font.pointSize()+2)
        self.answerText = QLabel("")
        # Need to add a QLayout
        # Want the branch name on top of the combo box option 
        self.binLabel = QLabel("Branch 1")
        self.binLabel.setStyleSheet("font-size: 25px;" "font:bold")
        self.binLayout.addWidget(self.binLabel, 1, 0, Qt.AlignBottom | Qt.AlignCenter) # where in the small table we add
        
        self.dropDown = QComboBox()
        self.dropDown.setFixedSize(300, 50)
        self.dropDown.setFont(font)
        self.dropDown.addItems(self.binValues)
        self.dropDown.setCurrentIndex(self.binIndices[0])
        self.dropDown.activated.connect(self.dropDownTextSelected)
        self.binLayout.addWidget(self.dropDown, 2, 0, Qt.AlignTop | Qt.AlignCenter)

        # Second dropdown box
        self.binLabel2 = QLabel("Branch 2")
        self.binLabel2.setStyleSheet("font-size: 25px;" "font:bold")
        self.binLayout.addWidget(self.binLabel2, 1, 1, Qt.AlignBottom | Qt.AlignCenter)
        
        self.dropDown2 = QComboBox()
        self.dropDown2.setFixedSize(300, 50)
        self.dropDown2.setFont(font)
        self.dropDown2.addItems(self.binValues)
        self.dropDown2.setCurrentIndex(self.binIndices[1])
        self.dropDown2.activated.connect(self.dropDownTextSelected)
        self.binLayout.addWidget(self.dropDown2, 2, 1, Qt.AlignTop | Qt.AlignCenter)

        # 3rd drop down
        self.binLabel3 = QLabel("Branch 3")
        self.binLabel3.setStyleSheet("font-size: 25px;" "font:bold")
        self.binLayout.addWidget(self.binLabel3, 1, 2, Qt.AlignBottom | Qt.AlignCenter)
        
        self.dropDown3 = QComboBox()
        self.dropDown3.setFixedSize(300, 50)
        self.dropDown3.setFont(font)
        self.dropDown3.addItems(self.binValues)
        self.dropDown3.setCurrentIndex(self.binIndices[2]) # SET THE VALUE TO PREVIOUS RESULTS
        self.dropDown3.activated.connect(self.dropDownTextSelected)
        self.binLayout.addWidget(self.dropDown3, 2, 2, Qt.AlignTop | Qt.AlignCenter)
       
        self.dropDownTextSelected("") # grab the current text in the dropdown menus

        self.submitButton = QPushButton("Submit") # Make a blank button
        self.submitButton.setStyleSheet("QPushButton {font-size: 25px;" "font:bold}\n"
                                        "QPushButton::pressed {background-color: #4F9153;}") # 
        # self.submitButton.clicked.connect(self.submitButtonClicked)
        self.submitButton.clicked.connect(self.submit)
        self.submitButton.setFixedSize(150, 50)
        self.binLayout.addWidget(self.submitButton, 3, 0)

        
        self.answerText.setStyleSheet("font-size: 25px;")
        self.binLayout.addWidget(self.answerText, 3, 1, 1, 2, Qt.AlignBottom | Qt.AlignCenter)       


    
    def dropDownTextSelected(self, _):
        self.interacted = True
        self.binAnswers[0] = self.dropDown.currentText()
        self.binIndices[0] = self.dropDown.currentIndex()

        self.binAnswers[1] = self.dropDown2.currentText()
        self.binIndices[1] = self.dropDown2.currentIndex()

        self.binAnswers[2] = self.dropDown3.currentText()
        self.binIndices[2] = self.dropDown3.currentIndex()

        print(f"Different text selected: {self.binAnswers}")

        self.answerText.setText("")
        self.answerText.setStyleSheet("font-size: 25px;")
    
    def setDropDownValues(self):
        self.dropDown.setCurrentIndex(self.binIndices[0])
        self.dropDown2.setCurrentIndex(self.binIndices[1])
        self.dropDown3.setCurrentIndex(self.binIndices[2])


    def submitButtonClicked(self):
        print("SUBMIT BUTTON SELECTED")
        if self.submitButton.isChecked():
            self.submitButton.setStyleSheet("background-color: #4F9153;")
        
        if self.screenType == "manipulation" or self.screenType == "bin" or self.screenType == "scale":
            self.previousScreen = self.screenType
            self.correctFeature = self.glWidgetTree.correctFeature
            self.index = self.glWidgetTree.index

        self.screenType = "submit"
        self.loadScreen() 

    
    def retryButtonClicked(self):
        self.retryButton.setStyleSheet("background-color: #4F9153;")
        print(f"RETRY BUTTON SELECTED {self.previousScreen}")
        self.screenType = self.previousScreen
        self.loadScreen()


    # Need to load the next immediate screen
    def nextPageButtonClicked(self):
        print("NEXT BUTTON SELECTED")

        # if self.nextButton.isChecked():
        #     self.nextLabel.setText("Loading Next Page")
        # Should only click if a) interacted with the screen, b) screenType is normal or c) answer is Correct
        
        # if self.screenType == "normal" or (self.interacted and self.screenType == "scale")  or (self.isCorrect and self.screenType == "manipulation") or (self.isCorrect and self.screenType == "bin"):
        if self.interacted and self.isCorrect:    
            # self.nextLabel.setText("Loading Next Page")
            self.pageIndex += 1 # increment the page by 1


            # If the screenType is "prune" then we need to save the user's cutSequenceDict from the glWidget
            # save the values under the tree name 
            if self.screenType == "draw_tutorial" or self.screenType == "prune":
                print(f"\nUser's data: {self.curTree[:-4]}")
                print(self.glWidgetTree.cutSequenceDict)

            


            if self.isCorrect:
                self.isCorrect = False

            if self.interacted:
                self.interacted = False
            
            self.nextButton.setEnabled(False)

            # Change the tree file (if needed)
            treeFile = self.curTree
            if self.curTree == self.trees[self.pageIndex]:
                treeFile = None
            
            # change the directory
            directory = self.directories[self.pageIndex]
            self.manipulationDir = directory
            if self.directories[self.pageIndex] == "":
                directory = None
                
            # Reload any new files as needed
            self.meshDictionary = self.load_mesh_files(treeFile=treeFile, 
                                                    branchFiles=self.jsonData["Manipulation Files"], 
                                                    manipDirectory=directory,            
                                                    skyBoxFile=None) # Only need to load the skybox once
            
            self.glWidgetTree.loadNewMeshFiles(self.meshDictionary) # Load the new mesh directory values
            self.screenType = self.layouts[self.pageIndex] # Set the next page index
            self.index = 0 # set index to 0
            self.loadScreen()
        else:
            text = "Cannot continue until 'Your Task' is completed"
            self.nextLabel.setText(text)
            self.nextLabel.setStyleSheet("font-size: 15px;")
            
    
    def compareBinAnswers(self, answers, values):
        for answer, value in zip(answers, values):
            if answer != value:
                return False 
        return True
    

    def labelButtonClicked(self):
        checked = True
        self.interacted = True
        self.nextButton.setEnabled(True)
        
        if self.screenType == "normal":
            self.isCorrect = True

        if self.labelButton.isChecked():
            self.labelButton.setText("Labels Off")
            checked = True 
        else:
            self.labelButton.setText("Labels On")
            checked = False
        self.glWidgetTree.addLabels(checked) # activate the label check
        

    def loadJSONWorkflow(self):
        workflowDict = JSONFile("workflow.json", "o").data
        
        # Get the different components
        testWorkflow = workflowDict["Test"]
        modules = []
        labels = []
        layout = []
        directories = []
        trees = []

        for pageInfo in testWorkflow:
            modules.append(pageInfo["Module"])
            labels.append(pageInfo["Your Task"])
            layout.append(pageInfo["Layout"])
            directories.append(pageInfo["File Directory"])
            trees.append(pageInfo["Tree File"])

        return modules, labels, layout, directories, trees


if __name__ == '__main__':

    app = QApplication(sys.argv)

    # fmt = QSurfaceFormat()
    # fmt.setDepthBufferSize(24)
    # # print(QCoreApplication.arguments())
    # QSurfaceFormat.setDefaultFormat(fmt)


    window = Window() # GLDemo()
    # window = MainWindow()
    window.show()
    # sys.exit(app.exec_())
    sys.exit(app.exec())