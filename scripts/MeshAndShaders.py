#!/usr/bin/env python3

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
        
        # Load and set the vertices
        self.load_mesh(self.load_obj)
        self.vertexFaces = self.get_mesh_vertex() # all vertices grouped by their faces
        # print(self.vertexFaces.size)s

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
        
    def get_mesh_vertex(self):
        faces = []
        for face in self.faces:
            vertices = []
            for vertex in face:
                vertices.append(self.mesh.vertices[vertex]) # extracts the vertex at the position in the obj file
            faces.append(vertices)
        return np.array(faces, dtype=np.float32)

    ###############################################
    # DESCRIPTION: Find a set of faces that exist within a given region defined by a line represented by point1 and point2
    # INPUT: 
    #   - point1: a floating point list representing the (x1, y1) point a person draws on their screen in world coordinates
    #   - point2: a floating point list representing the (x2, y2) point a person draws on their screen in world coordinates
    #   - modelMt: the 4x4 model matrix to convert vertex coordinates to world coordinates
    # OUTPUT:
    #   - intersectFaces: a list of faces that have values inside the bounded box of (x1, y1), (x1, y2), (x2, y1), (x2, y2).
    #                     Will return None if no face intersects the area
    ################################################
    def faces_in_area(self, pt1, pt2, model):
        # need to find points that are in between pts 1 and 2
        # pts will be in world spaces

        # Check which is the minimum 
        minU = np.min([pt1[0], pt2[0]])
        maxU = np.max([pt1[0], pt2[0]])
        minV = np.min([pt1[1], pt2[1]])
        maxV = np.max([pt1[1], pt2[1]])

        if maxU - minU < 0.1:
            minU -= 0.1
            maxU += 0.1
        elif maxV - minV < 0.1:
            minV -= 0.1
            maxV += 0.1

        print(f"Bounding box between ({minV}, {minV}) and ({maxV}, {maxV})")   

        intercepts = []
        worldFaces = self.convertToWorldCoord(model, self.vertexFaces)
        # uvdFaces = self.convertFacesToUVD(projection, view, model, self.vertexFaces)
        # Need to loop through each vertexFace
        for face in worldFaces:
            faceIntercept = False
            for worldVtx in face:                
                # check if inside bounding box
                if (worldVtx[0] >= minU) and (worldVtx[0] <= maxU) and (worldVtx[1] >= minV) and (worldVtx[1] <= maxV):
                    faceIntercept = True
        
            # check if the line is possibly inside a face as well:
            # mainly check if within the bounding box of the face
            # Need to check if one or both vertices are inside the 

            # see if the drawing point is inside another face's bounding box
            triangleMinU = np.min([face[0][0], face[1][0], face[2][0]])
            triangleMaxU = np.max([face[0][0], face[1][0], face[2][0]])
            triangleMinV = np.min([face[0][1], face[1][1], face[2][1]])
            triangleMaxV = np.max([face[0][1], face[1][1], face[2][1]])

            pt1Inside = ((pt1[0] >= triangleMinU) and (pt1[0] <= triangleMaxU)) and ((pt1[1] >= triangleMinV) and (pt1[1] <= triangleMaxV))
            pt2Inside = ((pt2[0] >= triangleMinV) and (pt2[0] <= triangleMaxV)) and ((pt2[1] >= triangleMinV) and (pt2[1] <= triangleMaxV))

            # as long as one point could be inside the triangle, then test
            if pt1Inside or pt2Inside:
                faceIntercept = True
            
            if faceIntercept:
                intercepts.append(face) # save the world faces  

        if len(intercepts) == 0:
            return None           

        return np.array(intercepts, dtype=np.float32)


    def intersect_faces(self, u1, v1, u2, v2, proj, view, model): 
        # need to find the set of faces that are 
        #   A) within the bounded bowx given by (x1, y1) (x1, y2) (x2, y2) (x2, y1)
        #   b) faces that contain either the point (x1, y1) or (x2, y2)
        
        # check if the line is near vertical or horizontal
        # if so, add a little more area to the surrounding area
        delta = 0.05
        minU = np.min([u1, u2])
        minV = np.min([v1, v2])
        maxU = np.max([u1, u2])
        maxV = np.max([v1, v2])

        if maxU - minU <= delta:
            print(f"Nearly vertical")
            minU -= delta
            maxU += delta
        
        if maxV - minV <= delta:
            print("Nearly horizontal")
            minV -= delta
            maxV += delta
        
        print(f"Bounded u,v area: ({minU}, {minV}), ({minU}, {maxV}), ({maxU}, {maxV}), ({maxU}, {minV})")

        # Convert faces to uvd coordinates
        uvdFaces = self.convertFacesToUVD(proj, view, model, self.vertexFaces)

        intersected = []
        vertices = []
        intersectCount = 0
        # Need to compare and see if any of the faces are in the same u, v, bounded region 
        for count, face in enumerate(uvdFaces):
            bound = self.in_bound(minU, minV, maxU, maxV, face)
            inside = self.in_face(u1, v1, u2, v2, face)
            if bound or inside:
                # intersected.append(face)
                vertices.append(count)
                intersectCount += 1

        # print("Intersect Face Count: ", intersectCount)
        # print("Intersected face vertices:", vertices)
        
        if intersectCount > 0:
            for v in vertices:
                intersected.append(self.vertexFaces[v])
            return np.array(intersected, dtype=np.float32)
        else:
            return None
        


    def in_bound(self, minU, minV, maxU, maxV, face):
        # checks to see if the face (or part of the face) is in the given bounds
        vertex1 = face[0]
        vertex2 = face[1]
        vertex3 = face[2]
        bound1 = (vertex1[0] >= minU and vertex1[0] <= maxU) and (vertex1[1] >= minV and vertex1[1] <= maxV)
        bound2 = (vertex2[0] >= minU and vertex2[0] <= maxU) and (vertex2[1] >= minV and vertex2[1] <= maxV)
        bound3 = (vertex3[0] >= minU and vertex3[0] <= maxU) and (vertex3[1] >= minV and vertex3[1] <= maxV)
        
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
        return uv1Inside or uv2Inside

    
    def convertFaceToWorld(self, model, face):
        worldFace = []
        for v in face:
            vertex = np.ones(4)
            vertex[:3] = v
            world = model @ np.transpose(vertex)
            worldFace.append(world[:3])
        return np.array(worldFace, dtype=np.float32)


    def convertFacesToUVD(self, projection, view, model, faces):
        uvdFaces = []
        for face in faces:
            f = []
            for v in face:
                # do the math to convert to world coordinates
                coord = np.ones(4)
                coord[:3] = v 
                vertex = projection @ view @ model @ coord
                vertex /= vertex[3]
                # append to list
                f.append(vertex[:3])        
            uvdFaces.append(f)
        return np.array(uvdFaces, dtype=np.float32) 


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

        self.CUBE_VERTEX_SHADER = """
        # version 330 core
        //layout (location = 0) in vec2 aTexCoord;
        layout (location = 2) in vec3 vertexPos; // location 2

        uniform mat4 projection;
        uniform mat4 view;

        out vec3 localPos;

        void main()
        {
            
            gl_Position = projection * view * vec4(vertexPos, 1.0);
            localPos = aPos
        }  
        """

        self.CUBE_FRAGMENT_SHADER = """
        # version 330 core
        in vec3 localPos;

        //uniform sampler2D equirectangularMap;

        const vec2 invAtan = vec2(0.15191, 0.3183);

        vec2 SampleSphericalMap(vec3 v)
        {
            vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
            uv *= invAtan;
            uv += 0.5;
            return uv;
        }

        void main()
        {   
            vec2 uv = SampleSphericalMap(normalize(localPos));
            vec3 color = texture(equirectangularMap, uv).rgb
            FragColor = vec4(color, 1.0);
        }        

        """
        

    def calculatePos(self):
        print(self.projection)
        print(self.view)
        for vertex in self.skyVertices:
            v = np.ones(4)
            v[:3] = vertex

            mvp = self.projection @ self.view @ np.transpose(vertex)
            mvp = mvp / mvp[3]
            print(mvp)

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
        

    

    def initializeGL(self):
        print(self.getGlInfo())
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        
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

        self.initializeSkyBox()



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
        # print(self.projection)

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

        # gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        # gl.glUseProgram(self.program) 
        # gl.glBindVertexArray(self.vao)

        # modelLoc = gl.glGetUniformLocation(self.program, "model")
        # gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, self.model) # self.rotation

        # projLoc = gl.glGetUniformLocation(self.program, "projection")
        # gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection)

        # viewLoc = gl.glGetUniformLocation(self.program, "view")
        # gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view)

        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, int(self.vertices.size / 3)) # int(self.vertices.size / 3)
        # gl.glBindVertexArray(0) # unbind the vao
        # gl.glPopMatrix()

        self.drawSkyBox()


    # def paintGL(self):
    #     gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    #     gl.glDisable(gl.GL_DEPTH_TEST)
    #     gl.glDisable(gl.GL_CULL_FACE)

    #     gl.glLoadIdentity()
    #     gl.glPushMatrix()

    #     gl.glUseProgram(0)

    #     # set last row and column to 0 so it doesn't affect translations but only rotations
    #     model = mt.create_identity()
    #     # model = mt.create_from_scale([41.421, 41.421, -100])
    #     model = mt.create_from_scale([0, 0, -5.3888])

    #     self.view = mt.create_identity()
    #     # view[:,3] = np.zeros(4)
    #     # view[3, :] = np.zeros(4)
    #     # view = np.zeros((4, 4))
    #     # view[:-1, :-1] = mt.create_identity()[:-1, :-1]

    #     # projection = np.transpose(mt.create_perspective_projection_matrix(45.0, self.width / float(self.height), 0.1, 100.0))

    #     # set the depth function to equal
    #     # oldDepthFunc = gl.glGetIntegerv(gl.GL_DEPTH_FUNC)
    #     gl.glDepthFunc(gl.GL_LEQUAL)
    #     gl.glDepthMask(gl.GL_FALSE)

    #     gl.glUseProgram(self.skyProgram)
    #     gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.skyTexture)
    #     gl.glBindVertexArray(self.skyVAO)

    #     modelLoc = gl.glGetUniformLocation(self.skyProgram, "model")
    #     gl.glUniformMatrix4fv(modelLoc, 1, gl.GL_TRUE, model) # self.rotation

    #     projLoc = gl.glGetUniformLocation(self.skyProgram, "projection")
    #     gl.glUniformMatrix4fv(projLoc, 1, gl.GL_TRUE, self.projection) # use the same projection values
    #     # print(np.transpose(mt.create_perspective_projection_matrix(45.0, self.width / float(self.height), 0.1, 10.0)))

    #     viewLoc = gl.glGetUniformLocation(self.skyProgram, "view")
    #     gl.glUniformMatrix4fv(viewLoc, 1, gl.GL_TRUE, self.view) # use the same location view values

    #     # self.calculatePos()


    #     # gl.glDrawArrays(gl.GL_LINES, 0, int(self.skyVertices.size))
    #     gl.glDrawElements(gl.GL_QUADS, int(self.skyVertices.size), gl.GL_UNSIGNED_INT, 0)
    #     # gl.glDrawElements(gl.GL_POINTS, 0, int(self.skyVertices.size), gl.GL_UNSIGNED_INT, 0)

    #     gl.glBindVertexArray(0) # unbind the vao
    #     gl.glPopMatrix()

    #     gl.glDepthFunc(gl.GL_LESS)
    #     # gl.glDepthFunc(oldDepthFunc) # set back to default value
    #     gl.glDepthMask(gl.GL_TRUE)
    #     gl.glUseProgram(0)
    #     gl.glEnable(gl.GL_DEPTH_TEST)
    #     gl.glEnable(gl.GL_CULL_FACE)


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
    app = QApplication(sys.argv)
    window = Window() # GLDemo()
    window.show()
    sys.exit(app.exec_())
