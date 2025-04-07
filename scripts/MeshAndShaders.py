#!/usr/bin/env python3

import sys
sys.path.append('../')
import os

from PySide6 import QtCore, QtGui, QtOpenGL

from PySide6.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QComboBox, QProgressBar
    # QOpenGLWidget

from PySide6.QtOpenGLWidgets import QOpenGLWidget

from PySide6.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint, QCoreApplication, QPoint

# from PySide6.QtOpenGL import QGLWidget, QGLContext

from PySide6.QtGui import QFont
# from PySide6.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4, QSurfaceFormat, QPainter,
from OpenGL.GL.shaders import compileShader, compileProgram

# from PyQt6 import QtCore      # core Qt functionality
# from PyQt6 import QtGui       # extends QtCore with GUI functionality
# from PyQt6 import QtOpenGL    # provides QGLWidget, a special OpenGL QWidget
from PIL import Image
from scripts.BranchGeometry import BranchGeometry     # personal class used to help participants manipulate branches

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo
import pywavefront
import numpy as np
import ctypes                 # to communicate with c code under the hood
import pyrr.matrix44 as mt
from threading import Thread
import time

from multiprocessing import Process, Value, Array
from freetype.raw import * # allows us to display text on the screen
from freetype import *

#########################################
#
# NAME: Shader
# DESCRIPTION: Class for handling opengl shaders. Specifically reading in and 
#               compiling shaders (found in the shaders folder) from source
########################################
class Shader:
    def __init__(self, shaderType: str, shaderFile: str):

        self.shader = None
        self.shaderType = shaderType # describing if vertex or fragment shader
        self.shaderFile = shaderFile 

        # shaderSource = self.shaders[shaderType]  # gets the source code from above
        if shaderType == "vertex":
            # self.shader = QOpenGLShader(QOpenGLShader.Vertex)
            self.shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        else:
            # self.shader = QOpenGLShader(QOpenGLShader.Fragment)
            self.shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        shaderPath = "../shaders/" + str(shaderFile)
        source = self.readShader(shaderPath)
        gl.glShaderSource(self.shader, source)
        gl.glCompileShader(self.shader)
        # isCompiled = self.shader.compileSourceFile(shaderPath)

        # if isCompiled is False:
        #     print(self.shader.log())
        #     raise ValueError(
        #         "{0} shader {2} known as {1} is not compiled".format(shaderType, shaderName, shaderPath)
        #     )
        
    def readShader(self, fname):
        with open(fname, 'r') as f:
            shader_source = f.read()
        return shader_source


class Mesh(QWidget):
    def __init__(self, load_obj = None):
        if load_obj is None:
            load_obj = "/obj_files/testMonkey.obj"
        else:
            self.load_obj = load_obj
        
        self.mesh = None
        self.vertices = None
        self.faces = None
        self.mesh_list = None
        self.normals = None
        self.vertexFaces  = None
        self.intersectCount = 0
        self.bound = False
        self.inside = True
        self.faceCount = 0
        self.clipFaces = []
        self.centerFaces = None
        # Load and set the vertices
        
        self.load_mesh(self.load_obj)
        self.vertexFaces, self.centerFaces = self.get_mesh_vertex() # all vertices grouped by their faces
        # print(self.vertexFaces.size)
        # self.normals, self.vertexPos, self.vertexFaces = self.split_vertices()
        # self.set_vertices()
    
    def load_mesh(self, fname = None):
        if fname is None:
            self.load_obj = self.init_obj
        else:
            self.load_obj = fname
        self.mesh = pywavefront.Wavefront(self.load_obj, collect_faces=True)
        self.mesh_list = self.mesh.mesh_list
        self.faces = self.mesh_list[0].faces
        # self.normals = self.mesh_list[0].normals
        # print(self.mesh_list[0].materials[0].vertices[:10])
        self.vertices = self.mesh_list[0].materials[0].vertices # gets the vertices in [nx, ny, nz, x, y, z] format
        self.count = len(self.mesh.vertices)


    #############################################
    # DESCRIPTION: write the participant's cuts to a mesh file 
    #   - save mesh file of cuts as PID_#_treeName.obj in a folder under PID_#
    #
    # INPUT:
    #   - treeName: a String representing the name of the .obj tree file participants interacted with (excluding .obj)
    #   - pid: an Int representing the participant's ID number 
    #   - cutDictionary: a List of dictionaries containing the sequence of cuts, a participant's cut decision,
    #                and vertices of the cut.
    # OUTPUT: None        
    #############################################
    def write_mesh_file(self, treeName, pid, cutDictionary):
        # Loop through dictionary containing: Cut decision (in order) and their vertices
        # Write the objects to the same mesh file 
        fname = "PID_" + str(pid) + "_" + treeName + ".obj"
        
        # loop over all the cuts and through their vertices to add the files
        # create the directory if it doesn't exist for the participant
        pid_directory = f"../user_data/PID_{pid}"

        # Create a directory for the PID if it doesn't exist
        if not os.path.exists(pid_directory):
            os.makedirs(pid_directory)
        

        file = pid_directory + "/" + fname

        with open(file, "a") as mesh:
            
            for idx in range(len(cutDictionary["Rule"])):
                objName = "Cut_" + str(idx+1) + "_" + cutDictionary["Rule"][idx]

                # loop over the vertices to generate the mesh file.
                # always 8 vertices in the data!!

                mesh.write(f"o {objName}\n") # tell what object it is first
                

                # vertices first obj has vertices 1-8
                for v in cutDictionary["Vertices"][idx]:
                    mesh.write(f"v {v[0]} {v[1]} {v[2]}\n")

                # modify face value by shifting it 8 for each new object in the mesh
                mesh.write(f"f {(1 + idx * 8)} {(2 + idx * 8)} {(3 + idx * 8)} {(4 + idx * 8)}\n")
                mesh.write(f"f {(5 + idx * 8)} {(8 + idx * 8)} {(7 + idx * 8)} {(6 + idx * 8)}\n")
                mesh.write(f"f {(1 + idx * 8)} {(5 + idx * 8)} {(6 + idx * 8)} {(2 + idx * 8)}\n")
                mesh.write(f"f {(2 + idx * 8)} {(6 + idx * 8)} {(7 + idx * 8)} {(3 + idx * 8)}\n")
                mesh.write(f"f {(3 + idx * 8)} {(7 + idx * 8)} {(8 + idx * 8)} {(4 + idx * 8)}\n")
                mesh.write(f"f {(5 + idx * 8)} {(1 + idx * 8)} {(4 + idx * 8)} {(8 + idx * 8)}\n")
                
    
                """ 
                Start object line with o Name
                Uses the same VN for all vertices
                vn 0.0000 -1.0000 0.0000
                vn 0.0000 1.0000 0.0000
                vn 1.0000 0.0000 0.0000
                vn -0.0000 -0.0000 1.0000
                vn -1.0000 -0.0000 -0.0000
                vn 0.0000 0.0000 -1.0000

                Cube Faces: v//vn

                 2________4
                 /|      /|
                /_|_____/ |
                1       3 |
                | |____|__|
                | 6    |  8
                | /    | /
                |/_____|/
                5       7
                
                # I DON'T INCLUDE NORMALS
                f 1//1 2//1 3//1 4//1
                f 5//2 8//2 7//2 6//2
                f 1//3 5//3 6//3 2//3
                f 2//4 6//4 7//4 3//4
                f 3//5 7//5 8//5 4//5
                f 5//6 1//6 4//6 8//6
                """


    def split_vertices(self):
        # print(self.vertices.size)
        # print(len(self.vertices) / 6)
        normals = []
        vertices = []
        faceVertices = []
        
        # want to split each into the normals and group vertices by
        for i in range(int(len(self.vertices) / 6)):   
            # split the vertices into normals and vertex
            index = 6*i
            # Extract the normals
            normal = self.vertices[index:index+3]
            normals.append(normal)

            # Extract the vertices and store as faces
            vertex = self.vertices[index+3:index+6]
            vertices.append(vertex)

        # split vertices into the triangle faces
        for i in range(int(len(vertices)/3)):
            faceVertices.append(vertices[3*i:3*i + 3])
        
        return normals, vertices, faceVertices

    # def set_vertices(self):
    #     self.vertices = np.array(self.mesh.vertices, dtype=np.float32)
    #     self.count = len(self.mesh.vertices)

    #     self.vertices = self.get_mesh_vertex()

    #     # listVertex = []
    #     # for v in self.mesh.vertices:
    #     #     listVertex.extend(v)
    #     # return np.array(listVertex, dtype=np.float32)

    # GET THE CENTER OF THE FACE
    def center_of_face(self, vertices):
        v1 = vertices[0]
        v2 = vertices[1]
        v3 = vertices[2]

        # look at the center point for each value
        cX = (v1[0] + v2[0] + v3[0]) / 3
        cY = (v1[1] + v2[1] + v3[1]) / 3
        cZ = (v1[2] + v2[2] + v3[2]) / 3
        
        return [cX, cY, cZ, 1]
        
    def get_mesh_vertex(self):
        faces = []
        centers = []
        for face in self.faces:
            # self.faceCount += 1
            vertices = []
            for vertex in face:
                vertices.append(self.mesh.vertices[vertex]) # extracts the vertex at the position in the obj file
            center = self.center_of_face(vertices)
            centers.append(center) # get the center of all vertices
            faces.append(vertices) # get all the faces
        return np.array(faces, dtype=np.float32), np.array(centers, dtype=np.float32)

    ###############################################
    # DESCRIPTION: Find a set of faces that exist within a given region defined by a line represented by point1 and point2
    # INPUT: 
    #   - point1: a floating point list representing the (x1, y1) point a person draws on their screen in world coordinates
    #   - point2: a floating point list representing the (x2, y2) point a person draws on their screen in world coordinates
    #   - modelMt: the 4x4 model matrix to convert vertex coordinates to world coordinates
    # OUTPUT:
    #   - intersectFaces: a list of faces that have values inside the bounded box of (x1, y1), (x1, y2), (x2, y1), (x2, y2).
    #                     Will return None if no face intersects the area
    ###############################################
    def faces_in_area(self, x1, y1, x2, y2):
        # need to find points that are in between pts 1 and 2
        # pts will be in world spaces

        start = time.time()
        # Check which is the minimum 
        minX = np.min([x1, x2])
        maxX = np.max([x1, x2])
        minY = np.min([y1, y2])
        maxY = np.max([y1, y2])

        if maxX - minX < 0.1:
            minX-= 0.1
            maxX += 0.1
        elif maxX - minX < 0.1:
            minX -= 0.1
            maxX += 0.1

        # print(f"Bounded u,v area: ({minX}, {minY}), ({minX}, {maxY}), ({maxX}, {maxY}), ({maxX}, {minY})")  

        xRows = self.centerFaces[:, 0]
        yRows = self.centerFaces[:, 1]

        xBound = np.where(xRows >= minX, True, False) & np.where(xRows <= maxX, True, False)
        yBound = np.where(yRows >= minY, True, False) & np.where(yRows <= maxY, True, False)
        
        inBound = xBound & yBound
        intersect = self.vertexFaces[inBound]
        
        # find intersects in the same area
        # for face in self.centerFaces:
            # Look for values in bound

            # bound = self.in_bound(minX, minY, maxX, maxY, face)
            # inside = self.in_face(x1, y1, x2, y2, face)
           
            # if bound or inside:
            #     intersect.append(face)
        
        # print(f"Intersect with local vertices time: {time.time() - start}")
        return np.array(intersect, dtype=np.float32)

   

   

    def intersect_faces(self, u1, v1, u2, v2, projection, view, model): 
        # need to find the set of faces that are 
        #   A) within the bounded bowx given by (x1, y1) (x1, y2) (x2, y2) (x2, y1)
        #   b) faces that contain either the point (x1, y1) or (x2, y2)
        
        # check if the line is near vertical or horizontal
        # if so, add a little more area to the surrounding area
        start = time.time()
        # self.intersectFaces = [] # reset to empty
        # self.intersectCount = 0
        
        delta = 0.05
        minU = np.min([u1, u2])
        minV = np.min([v1, v2])
        maxU = np.max([u1, u2])
        maxV = np.max([v1, v2])

        if maxU - minU <= delta:
            # print(f"Nearly vertical")
            minU -= delta
            maxU += delta
        
        if maxV - minV <= delta:
            # print("Nearly horizontal")
            minV -= delta
            maxV += delta
        
        # print(f"Bounded u,v area: ({minU}, {minV}), ({minU}, {maxV}), ({maxU}, {maxV}), ({maxU}, {minV})")

        # Convert faces to uvd coordinates
        # self.clipFaces, vertexID = self.convertFacesToUVD(proj, view, model, self.vertexFaces)
        centerClipFaces = self.convertFacesToUVD(projection, view, model, self.centerFaces)
        # intersected = []
        # vertices = []
        # intersectCount = 0

        xRows = centerClipFaces[:, 0]
        yRows = centerClipFaces[:, 1]

        xBound = np.where(xRows >= minU, True, False) & np.where(xRows <= maxU, True, False)
        yBound = np.where(yRows >= minV, True, False) & np.where(yRows <= maxV, True, False)

        inBound = xBound & yBound
        if not np.any(inBound):
            return None
        else:
            return self.vertexFaces[inBound]
        

        

    def doesIntersect(self, minU, minV, maxU, maxV, u1, v1, u2, v2, face, localFace):
        if self.in_bound(minU, minV, maxU, maxV, face) or self.in_face(u1, v1, u2, v2, face):
            self.intersectFaces.append(localFace)
            self.intersectCount += 1


    def in_bound(self, minU, minV, maxU, maxV, face):
        # checks to see if the face (or part of the face) is in the given bounds
        vertex1 = face[0]
        vertex2 = face[1]
        vertex3 = face[2]
        bound1 = (vertex1[0] >= minU and vertex1[0] <= maxU) and (vertex1[1] >= minV and vertex1[1] <= maxV)
        bound2 = (vertex2[0] >= minU and vertex2[0] <= maxU) and (vertex2[1] >= minV and vertex2[1] <= maxV)
        bound3 = (vertex3[0] >= minU and vertex3[0] <= maxU) and (vertex3[1] >= minV and vertex3[1] <= maxV)
        
        # if bound1 or bound2 or bound3:
        #     self.bound = True 
        return bound1 or bound2 or bound3
    

    def in_face(self, u1, v1, u2, v2, face):
        # checks to see if one of the points drawn 
        # get the bounded box for the points in the face
        minFaceU = np.min([face[0][0], face[1][0], face[2][0]])
        maxFaceU = np.max([face[0][0], face[1][0], face[2][0]])   
        minFaceV = np.min([face[0][1], face[1][1], face[2][1]])
        maxFaceV = np.max([face[0][1], face[1][1], face[2][1]])  

        uv1Inside = (u1 >= minFaceU and u1 <= maxFaceU) and (v1 >= minFaceV and v1 <= maxFaceV) 
        uv2Inside = (u2 >= minFaceU and u2 <= maxFaceU) and (v2 >= minFaceV and v2 <= maxFaceV)
        uv3Inside = (u2 >= minFaceU and u2 <= maxFaceU) and (v1 >= minFaceV and v1 <= maxFaceV)
        uv4Inside = (u1 >= minFaceU and u1 <= maxFaceU) and (v2 >= minFaceV and v2 <= maxFaceV) 

        # if uv1Inside or uv2Inside:
        #     self.inside = True
        return uv1Inside or uv2Inside or uv3Inside or uv4Inside

    
    def convertFaceToWorld(self, model, face):
        worldFace = []
        for v in face:
            vertex = np.ones(4)
            vertex[:3] = v
            world = model @ np.transpose(vertex)
            worldFace.append(world[:3])
        return np.array(worldFace, dtype=np.float32)


    #############################################################
    #  DESCRIPTION: Returns faces in uvd coordinates and determines which faces that are visible in the clip region i.e., 
    #               u & v in [-1, 1] and d in [0, 1] based on the projection, view, and model matrices.
    #
    #  INPUT:
    #       - projection: 4x4 numpy matrix representing the projection matrix from GLWidget
    #       - view: 4x4 numpy matrix representing the view matrix from GLWidget
    #       - model: 4x4 numpy matrix representing the model matrix from GLWidget
    #
    #  OUTPUT:
    #       - faces: a numpy array of faces with values inside the clip region
    ##############################################################

    def convertFacesToUVD(self, projection, view, model, faces):

        # print("Converting faces")
        start = time.time()
        self.faceCount = 0
        # vertexVisible = []
        # vertexID = []

        clipFaces = projection @ view @ model @ np.transpose(faces)
        clipFaces /= clipFaces[3] # divide by w

        clipFaces = np.transpose(clipFaces)
        # for id, face in enumerate(faces):
        #     if id < 5:
        #         print("Looping through faces")
        #     f = []
        #     vertexVisible = []
        #     for v in face:
        #         notVisible = False
        #         # do the math to convert to world coordinates
        #         coord = np.ones(4)
        #         coord[:3] = v 
        #         vertex = projection @ view @ model @ coord
        #         vertex /= vertex[3]

        #         # check to see if in the clip region
        #         if np.any(vertex > 1) or np.any(vertex < -1) or vertex[2] < 0:
        #             notVisible = True
        #         vertexVisible.append(notVisible)
        #         # append to list
        #         f.append(vertex[:3])         
        #     # uvdFaces.append(f)
        #     # only append to the list if the vertex is visible
        #     if not np.any(vertexVisible):
        #         clipFaces.append(f)
        #         vertexID.append(id)
        #         self.faceCount += 1
        
        # print(f"End of conversion to UVD: {time.time() - start}")

        return np.array(clipFaces, dtype=np.float32)


    def convertUVDtoWorld(self, projection, view, faces):
        worldFace = []
        for f in faces:
            face = []
            for vertex in f:
                clip = np.ones(4)
                clip[:3] = vertex
                eye = np.linalg.inv(projection) @ clip
                world = np.linalg.inv(view) @ eye
                world /= world[3]
                face.append(world[:3]) # only want the first 3 values x, y, z
            worldFace.append(face)
        return np.array(worldFace, dtype=np.float32)
                

    def convertToWorldCoord(self, modelMt, faces):
        worldFaces = []
        for face in faces:
            f = []
            for v in face:
                # do the math to convert to world coordinates
                coord = np.ones(4)
                coord[:3] = v 
                vertex = modelMt @ coord
                # append to list
                f.append(vertex[:3])        
            worldFaces.append(f)
        
        return np.array(worldFaces, dtype=np.float32)
                 


    def set_translate(self, x, y, z):
        gl.glTranslate(x, y, z)
    
    def set_scale(self, sx, sy, sz):
        gl.glScale(sx, sy, sz)
    
    def set_rotation(self, degrees, rx, ry, rz):
        gl.glRotate(degrees, rx, ry, rz)



class Character:
    def __init__(self, texID, size, bearing, advance):
        self.texID = texID
        self.size = size  # a tuple storing the size of the glyph
        self.bearing = bearing # a tuple describing the bearing of the glyph
        self.advance = advance



class TestMeshOpenGL(QOpenGLWidget):

    def __init__(self, parent=None):
        QOpenGLWidget.__init__(self, parent)

        self.mesh = Mesh("../obj_files/testMonkey.obj")
        self.vertices = np.array(self.mesh.vertices, dtype=np.float32)
        self.vao = None
        self.vbo = None
        self.texture = None
        # # self.mesh = Mesh("../obj_files/exemplarTree.obj")
        # self.vertices = np.array(self.mesh.vertices, dtype=np.float32)

        # GETTING THE BACKGROUND SKY BOX
        self.skyMesh = Mesh("../obj_files/skyBox.obj")
        self.skyVertices = np.array(self.skyMesh.vertices, dtype=np.float32)

        
        self.skyProgram = None
        self.skyTexture = None
        self.skyVAO = None
        self.skyVBO = None
        # self.skyMapDir = "../textures/skymap/"
        self.cubeTypes = [gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                          gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                          gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 
                          gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                          gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                          gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]
        
        # self.cubeFaces = ["px.png", # right
        #                   "nx.png", # left
        #                   "py.png", # top
        #                   "ny.png", # bottom
        #                   "pz.png", # front (flipped from pz)
        #                   "nz.png"] # back (flipped from nz)

        self.skyMapDir = "../textures/skymap/"        
        self.cubeFaces = ["px.png", # right
                          "nx.png", # left
                          "py.png", # top
                          "ny.png", # bottom
                          "pz.png", # front
                          "nz.png"] # back
        
        self.hdrTexture = None
        self.hdrVAO = None
        self.hdrVBO = None
        self.hdrProgram = None

        self.characters = {}
        self.textVAO = None
        self.textVBO = None
        self.textProgram = None
        self.textVertices = np.zeros((6, 4), dtype=np.float32)


        self.SIMPLE_VERTEX_SHADER = """
        # version 330 core
        //layout (location = 0) in vec2 aTexCoord;
        layout (location = 2) in vec3 vertexPos; // location 2

        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;

        out vec4 color;
        //out vec2 TexCoord;
        out vec3 TexCoord;

        void main()
        {
            //TexCoord = aTexCoord;
            //color = vec4(0.5, 0.25, 0.0, 1.0); 
            //gl_Position = projection * view * model * vec4(vertexPos, 1.0);

            vec4 pos = projection * view * * model * vec4(vertexPos, 1.0);
            //pos = view * vec4(vertexPos, 1.0);
            //gl_Position = pos;
            gl_Position = pos.xyww;
            TexCoord = vertexPos;
            //TexCoord = vec3(pos.x, pos.y, -pos.z);
        }  
        """

        self.SIMPLE_FRAGMENT_SHADER = """
        # version 330 core
        in vec4 color;
        //in vec2 TexCoord; // normal object textures
        in vec3 TexCoord; // for the cubeMap

        //uniform sampler2D ourTexture;
        uniform samplerCube ourTexture;

        //out vec4 treeColor;
        out vec4 FragColor;

        void main()
        {   
            //FragColor = vec4(0.56, 0.835, 1.0, 1.0);
            FragColor = texture(ourTexture, TexCoord);
            //treeColor = color * texture(ourTexture, TexCoord);
            //treeColor = color;
        }        

        """

        self.TEXT_VERTEX_SHADER = """
        # version 330 core
        layout (location = 0) in vec4 vertex;
        out vec2 TexCoords;

        uniform mat4 projection;

        void main()
        {
            gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
            TexCoords = vertex.zw;
        }
        """

        self.TEXT_FRAGMENT_SHADER = """
        # version 330 core
        in vec2 TexCoords;
        out vec4 color;

        uniform sampler2D text;

        void main()
        {
            vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
            color = vec4(1.0, 0.0, 0.0, 1.0) * sampled;
        }
        """


    def calculatePos(self):
        # print(self.projection)
        # print(self.view)
        for vertex in self.skyVertices:
            v = np.ones(4)
            v[:3] = vertex

            mvp = self.projection @ self.view @ np.transpose(vertex)
            mvp = mvp / mvp[3]
            # print(mvp)

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
        # angle = self.angle_to_radians(self.turntable)
        # rotation = mt.create_from_y_rotation(angle)
        model = scale # for rotating the modelView

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


    def angle_to_radians(self, angle):
        return angle * (np.pi / 180.0)


    def loadHDRImage(self, fname):        
        fname = str("../textures/") + str(fname)
        # LOAD IN THE TEXTURE IMAGE AND READ IT IN TO OPENGL
        texImg = Image.open(fname)
        # texImg = texImg.transpose(Image.FLIP_TO_BOTTOM)
        texData = texImg.convert("RGB").tobytes()

        self.hdrTexture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.hdrTexture)

        # need to load the texture into our program
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 
                        0,                      # mipmap setting (can change for manual setting)
                        gl.GL_RGB16F,              # texture file storage format
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
        # gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)


    def initHDRCubeMap(self):
        self.captureFBO = gl.glGenFramebuffers(1)
        self.captureRBO = gl.glGenRenderbuffers(1)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.captureFBO)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self.captureRBO)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_ATTACHMENT, self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, self.captureRBO)

        self.envCubeMap = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.envCubeMap)
        for i in range(6):
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.GL_RGB16F, self.width, self.height, 0, gl.GL_RGB, gl.GL_FLOAT, None)

        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR) 
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        


    #########################################################
    # DESCRIPTION: Initializing textures for loading text on the screen
    # Code inspired by: https://learnopengl.com/In-Practice/Text-Rendering
    ########################################################
    def initializeText(self):
        # Create a text program
        gl.glUseProgram(0)
        vertexShader = Shader("vertex", "text_shader.vert").shader # get out the shader value    
        fragmentShader = Shader("fragment", "text_shader.frag").shader
        # vertexShader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        # gl.glShaderSource(vertexShader, 1, str(self.TEXT_VERTEX_SHADER), None)
        # gl.glCompileShader(vertexShader)

        # fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        # gl.glShaderSource(fragmentShader, 1, self.TEXT_FRAGMENT_SHADER, None)
        # gl.glCompileShader(fragmentShader)

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
    
        # FT_Done_Face(face)
        # FT_Done_FreeType(ft)
        # create and bind the vertex attribute array
        self.textVAO = gl.glGenVertexArrays(1)
        # bind the vertex buffer object
        self.textVBO = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.textVAO)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.textVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.textVertices.itemsize * 6 * 4, self.textVertices, gl.GL_DYNAMIC_DRAW) 
        gl.glEnableVertexAttribArray(0) # set location = 0 in text vertex shader
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 4 * self.textVertices.itemsize, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)



    def renderText(self, text, x, y, scale, color):
        # gl.glEnable(gl.GL_CULL_FACE)
        gl.glUseProgram(0)
        gl.glLoadIdentity()
        gl.glPushMatrix()
        gl.glUseProgram(self.textProgram) # activate the program
        # allows us to map the text on screen using screen coordinates
        # textProject = np.transpose(mt.create_orthogonal_projection_matrix(0.0, self.width, 0.0, self.height, 0.1, 10.0))
        # print(textProject)
        textProjLoc = gl.glGetUniformLocation(self.textProgram, "projection")
        gl.glUniformMatrix4fv(textProjLoc, 1, gl.GL_TRUE, self.projection)

        viewLoc = gl.glGetUniformLocation(self.textProgram, "view")
        gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view)
        
        model = np.transpose(mt.create_from_translation([0, 0, -5]))
        modelLoc = gl.glGetUniformLocation(self.textProgram, "model")
        gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model)

        # colorLoc = gl.glGetUniformLocation(self.textProgram, "color")
        # gl.glUniform3fv(colorLoc, color[0], color[1], color[2])

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self.textVAO)
        
        for char in text:
            print(char)
            # intChar = ord(char) # convert the character to a number
            character = self.characters[char]

            xpos = x + character.bearing[0] * scale # get the x value from the bearing
            ypos = y - (character.size[1] - character.bearing[1]) * scale # get the y value from bearing and scale

            w = character.size[0] * scale
            h = character.size[1] * scale

            self.textVertices = np.array([[xpos,     ypos + h, 0.0, 0.0],
                                        [xpos,     ypos,     0.0, 1.0],
                                        [xpos + w, ypos,     1.0, 1.0],
                                        [xpos,     ypos + h, 0.0, 0.0],
                                        [xpos + w, ypos,     1.0, 1.0],
                                        [xpos + w, ypos + h, 1.0, 0.0]], dtype=np.float32)

            # Bind the character's texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, character.texID)
            # Update the buffer
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.textVBO)
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self.textVertices.nbytes, self.textVertices)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            # gl.glNamedBufferSubData(self.textVBO, 0, self.textVertices.nbytes, self.textVertices)
            # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6) # number of rows for text
            x += (character.advance >> 6) * scale
        
        gl.glBindVertexArray(0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glPopMatrix()
        gl.glUseProgram(0)
        # gl.glDisable(gl.GL_CULL_FACE)

    

    def initializeGL(self):
        print(self.getGlInfo())
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND) # For rendering text
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA) # For rendering text
        # gl.glDisable(gl.GL_CULL_FACE)

        self.initializeText()
        
        # # TREE SHADER
        # vertexShader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        # gl.glShaderSource(vertexShader, self.SIMPLE_VERTEX_SHADER)
        # gl.glCompileShader(vertexShader)
    
        # fragmentShader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        # gl.glShaderSource(fragmentShader, self.SIMPLE_FRAGMENT_SHADER)
        # gl.glCompileShader(fragmentShader)

        # self.program = gl.glCreateProgram()
        # gl.glAttachShader(self.program, vertexShader)
        # gl.glAttachShader(self.program, fragmentShader)
        # gl.glLinkProgram(self.program)

        # gl.glUseProgram(self.program)

        # # create and bind the vertex attribute array
        # self.vao = gl.glGenVertexArrays(1)
        # gl.glBindVertexArray(self.vao)

        # # bind the vertex buffer object
        # self.vbo = gl.glGenBuffers(1)
        # gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # # Bind the texture and set our preferences of how to handle texture
        # self.texture = gl.glGenTextures(1)
        # # print(f"texture ID {self.texture}")
        # gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture) 
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT) # will repeat the image if any texture coordinates are outside [0, 1] range for x, y values
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST) # describing how to perform texture filtering (could use a MipMap)
        # gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # # LOAD IN THE TEXTURE IMAGE AND READ IT IN TO OPENGL
        # texImg = Image.open("../textures/bark.jpg")
        # texData = texImg.convert("RGB").tobytes()

        # # need to load the texture into our program
        # gl.glTexImage2D(gl.GL_TEXTURE_2D, 
        #                 0,                      # mipmap setting (can change for manual setting)
        #                 gl.GL_RGB,              # texture file storage format
        #                 texImg.width,           # texture image width
        #                 texImg.height,          # texture image height
        #                 0,                      # 0 for legacy 
        #                 gl.GL_RGB,              # format of the texture image
        #                 gl.GL_UNSIGNED_BYTE,    # datatype of the texture image
        #                 texData)                # data texture 

        # # Reshape the list for loading into the vbo 
        # gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        # # gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex.nbytes, self.vertex, gl.GL_STATIC_DRAW)

        # # SET THE ATTRIBUTE POINTERS SO IT KNOWS LOCATIONS FOR THE SHADER
        # stride = self.vertices.itemsize * 8 # times 8 given there are 8 vertices across texture coords, normals, and vertex positions before next set
        # print(stride)
        # # TEXTURE COORDINATES
        # gl.glEnableVertexAttribArray(0)
        # gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0)) 

        # # NORMAL VECTORS (aNormal)
        # gl.glEnableVertexAttribArray(1)     # SETTING THE NORMAL
        # # gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(0)) # for location = 0
        # gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8)) # starts at position 2 with size 4 for float32 --> 2*4

        # # VERTEX POSITIONS (vertexPos)
        # gl.glEnableVertexAttribArray(2) # SETTING THE VERTEX POSITIONS
        # # gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, self.vertices.itemsize * 6, ctypes.c_void_p(12)) # for location = 1
        # gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(20)) # starts at position 5 with size 4 --> 5*4

        # self.initializeSkyBox()



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
                           100.0)   # z far
        self.projection = np.array(gl.glGetDoublev(gl.GL_PROJECTION_MATRIX))
        self.projection = np.transpose(self.projection)
        print(self.projection)

        # self.projection = mt.create_perspective_projection_matrix(45.0, aspect, 0.1, 10.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self): 
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glLoadIdentity()
        gl.glPushMatrix()

        translation = np.transpose(mt.create_from_translation([0, 0, -5.883])) # 0, -2, -5.883

        self.model = mt.create_identity()
        self.model = translation # @ x_rotation  # x-rotation only needed for some files

        # CALCULATE NEW VIEW (at our origin (0, 0, 0))
        self.view = mt.create_identity() # want to keep the camera at the origin (0, 0, 0) 

        gl.glUseProgram(0)
        # color = [1.0, 0.0, 0.0]
        


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


class Window(QMainWindow):

    def __init__(self, parent=None):
        # super(Window, self).__init__()
        QMainWindow.__init__(self, parent)
        # QtOpenGL.QMainWindow.__init__(self)
        self.resize(700, 700)
        self.setWindowTitle("TEST MESH WINDOW")

        self.central_widget = QWidget() # GLWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        
        self.glWidget = TestMeshOpenGL()
        self.layout.addWidget(self.glWidget)
        
        # self.glWidget = Test()
        # self.setCentralWidget(self.glWidget)

        # self.layout = QGridLayout(self.central_widget)





if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # window = Window() # GLDemo()
    # window.show()
    # sys.exit(app.exec_())

    circle = Mesh("../obj_files/circle.obj", circle=True)
    # print(vertices)


    # failY = [0.0, 1.1, 0.0, 1]
    # failY2 = [0.0, -1.1, 0.0, 1]
    # failX = [1.1, 0.0, 0.0, 1]
    # failX2 = [-1.1, 0.0, 0.0, 1]
    # failZ = [0.0, 0.0, 1.1, 1]
    # failZ2 = [0.0, 0.0, -0.1, 1]
    # passVal = [0.5, 0.5, 0.5, 1]
    # passVal2 = [0.0, 0.0, 0.0, 1]
    # passVal3 = [1.0, 1.0, 1.0, 1]
    # passVal4 = [-1.0, -1.0, 1.0, 1]

    # face1 = np.array([passVal, failX, failX2, passVal2, failY, failY2, passVal3, failZ, failZ2], dtype=np.float32)
    # projection = np.array([[2.41421366, 0, 0, 0],
    #                         [0, 2.41421366, 0, 0],
    #                         [0, 0, -1.02020204, -0.2020202],
    #                         [0, 0, -1, 0]], dtype=np.float32)
    
    # model = np.array([[1, 0, 0, -0.25],
    #                 [0, 1, 0, -2],
    #                 [0, 0, 1, -1.5],
    #                 [0, 0, 0, 1]], dtype=np.float32)



    # # print(faces.shape)
    # xRows = face1[:, 0]
    # yRows = face1[:, 1]
    # zRows = face1[:, 2]


    # yBound1 = np.where(yRows <= 1, True, False)
    # yBound2 = np.where(yRows >= -1, True, False)
    # zBound1 = np.where(yRows <= 1, True, False)
    # zBound2 = np.where(yRows >= 0, True, False)


    # xBound = np.where(xRows <= 1, True, False) & np.where(xRows >= -1, True, False)
    # yBound = np.where(yRows <= 1, True, False) & np.where(yRows >= -1, True, False)
    # zBound = np.where(zRows <= 1, True, False) & np.where(zRows >= 0.0, True, False)
    
    # # inBound = np.logical_and(zBound, np.logical_and(xBound, yBound, dtype=bool))
    # inBound = xBound & yBound & zBound

    # print(inBound)
    # if np.any(inBound):
    #     print("Face in bound")

    # print(np.transpose(face1[inBound]))
    # test = projection @ model @ np.transpose(face1[inBound])

    # print(test)
    # print(test / test[3])
    # test = np.any(faces, axis = 2, where = faces < -1)
    # test2 = np.any(faces, axis=2, where=faces > 1)

    # for i, face in enumerate(faces):
    #     for vertex in face:
    #         if np.any(vertex > 1) or np.any(vertex < -1) or vertex[2] < 0:
    #             print(f"vertex {vertex} not in clip range")
    #         else:
    #             print(f"Vertex {vertex} in clip range")

    


        

   

