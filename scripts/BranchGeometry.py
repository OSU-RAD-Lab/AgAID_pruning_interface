#!/usr/bin/env python3

import numpy as np
from json import load
import random
import ctypes

import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R
from geomdl import BSpline, operations, utilities     # https://nurbs-python.readthedocs.io/en/latest/module_bspline.html
# import geomdl.knotvector as kv  # https://nurbs-python.readthedocs.io/en/latest/module_knotvector.html

from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4
from shiboken2 import VoidPtr
#######################################################################
# Inspiration taken from make_branch_geometry.py in the treefitting repository
# Utilize the BSpline class 
#######################################################################

NUM_CTRL_PTS = 3                                        # Constant to record number of control pts (pt1, pt2, pt3) we have in 3D space

class BranchGeometry:

    _profiles = None
    _type_names = {"primary", "secondary", "tertiary"}      # explains if trunk, secondary, or tertiary branch
    _bud_shape = None
    

    def __init__(self, dir):

        # call global params
        BranchGeometry._read_profiles(dir)
        BranchGeometry._make_bud_shape()

        self.dir = dir
        # Initialize the control points for the branch geometry
        # Used when calculating the BSpline curve of the branch
        self.pt1 = [-0.5, 0.0, 0.0]
        self.pt2 = [0.0, 0.0, 0.0]
        self.pt3 = [0.5, 0.0, 0.0]

        # set values for the curve
        # More about B-Splines here: https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html
        # More information about knot vectors is here: https://saccade.com/writing/graphics/KnotVectors.pdf
        self.curve = BSpline.Curve()
        self.curve.degree = 2                                    # the basis function order is 3D
        self.curve.ctrlpts = [self.pt1, self.pt2, self.pt3]      # set the control points
        self.knotvector = utilities.generate_knot_vector(self.curve.degree, self.curve.ctrlpts_size) # have to have (degrees + # control pts + 1) knots
        # print("Knot vector", self.knotvector)
        self.curve.knotvector = self.knotvector
     
        self.curve.delta = 0.05

        self.curvePts = self.curve.evalpts     # evaluats the curve based on the ctrl points, degree, knot vector, and delta
        self.curve.evaluate()                  # evaluate the curve

        # information for the branch geometry
        self.startJunction = True           # flare the bottom to "connect" to side branch
        self.terminalBud = True             # make the end of the branch a terminal bud\
        self.startBud = 0.7                 # position of the starting bud
        self.budAngle = 0.8 * np.pi / 2     # the angle of the bud on the branch
        self.budLength = 0.1                # how long the bud is

        # Values people can modify
        self.numBuds = 3                    # Number of buds to add to the tree branch
        self.budObjects = {}                # Storing the bud objects to write to a mesh file for the branch
        self.budLocations = {}              # Where the bud is stored on the branch (position, angle)
        self.budSpacing = 0.1               # how much space is between each bud (want evenly spaced)  
        self.branchCurve = 0.1              # how much curve the branch will have 
        self.branchLength = 0.3             # the length of the bud
        self.startR = 0.03                   # the starting radius of the branch
        self.endR = self.startR/2           # the end radius of the branch

        # per branch/trunk parameters
        self.along = 10
        self.around = 64                    

        self.vertexLocs = np.zeros((self.along, self.around, 3)) # possible places the buds can be along the branch

        self.vertices = []
        self.faces = []

        # creates a branch based on the initial values so far 
        self.create_branch()


    @staticmethod
    def _read_profiles(dir):
        if BranchGeometry._profiles is not None:
            return 
        
        # create json files to store values for each type of branch
        BranchGeometry._profiles = {}
        for t in BranchGeometry._type_names:
            try:
                fname = dir + "/" + t + "_profiles.json"
                with open(fname, "r") as fp:
                    BranchGeometry._profiles[t] = load(fp)
            except:
                pass

    
    @staticmethod
    def _make_bud_shape():
        if BranchGeometry._bud_shape is None:
            totalPoints = 10
            BranchGeometry._bud_shape = np.zeros((2, totalPoints))
            BranchGeometry._bud_shape[0, :] = np.linspace(0, 1.0, totalPoints)
            BranchGeometry._bud_shape[0, -2] = 0.5 * BranchGeometry._bud_shape[0, -2] + 0.5 * BranchGeometry._bud_shape[0, -1]
            BranchGeometry._bud_shape[1, 0] = 1.0
            BranchGeometry._bud_shape[1, 1] = 0.95
            BranchGeometry._bud_shape[1, 2] = 1.05
            BranchGeometry._bud_shape[1, 3] = 1.1
            BranchGeometry._bud_shape[1, 4] = 1.05
            BranchGeometry._bud_shape[1, 5] = 0.8
            BranchGeometry._bud_shape[1, 6] = 0.7
            BranchGeometry._bud_shape[1, 7] = 0.5
            BranchGeometry._bud_shape[1, 8] = 0.3
            BranchGeometry._bud_shape[1, 9] = 0.0

    def num_vertices(self):
        return self.along * self.around
    
    def set_dimension(self, along=10, around=64):
        self.along = along
        self.around = around
        self.vertexLocs = np.zeros((self.along, self.around, 3))
    
    def set_ctrl_pts(self, pt1, pt2, pt3):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3

        # reset the curve ctrl pts
        self.curve.ctrlpts = [self.pt1, self.pt2, self.pt3]
        self.curvePts = self.curve.evalpts
        self.curve.evaluate()

        # self.create_branch()       # update faces and vertices for branch

    
    def pts_from_tangent(self, pt1, vec1, pt3):
        mid = np.array(pt1) + np.array(vec1) * 0.5
        self.set_ctrl_pts(pt1, mid, pt3)

    ##################################################################################
    #
    #   DESCRIPTION: Sets the branch radius of the branch at the beginning and end
    #   also sets if the branch is a junction point and has a terminal bud
    #
    ##################################################################################  
    def set_branch_radius(self, startR=1.0, endR=1.0, startJunction=True, terminalBud=False):
        self.startR = startR
        self.endR = endR
        self.startJunction = startJunction
        self.terminalBud = terminalBud


    def set_length(self, length):
        self.branchLength = length

    
    def set_branch_curve(self, curve):
        self.branchCurve = curve
        

    #######################################################
    # DESCRIPTION: returns the parametric value t on the Bezier curve based on the control pts
    #              pt1*(1-t)^2 + 2(1-t)t*pt2 + pt3*t^2
    # INPUT: 
    #   - t: value between [0, 1] 
    # OUTPUT:
    #   - set of 3D pts 
    #######################################################
    def pt_on_curve(self, t):
        pts = np.zeros(NUM_CTRL_PTS)
        for i in range(NUM_CTRL_PTS):
            pts[i] = self.pt1[i] * (1 - t) ** 2 + 2 * (1 - t) * t * self.pt2[i] + t ** 2 * self.pt3[i]

        return pts.transpose()
        
    def tangent_axis(self, t):
        vectors = np.zeros(NUM_CTRL_PTS)
        for i in range(NUM_CTRL_PTS):
            vectors[i] = 2 * t (self.pt1[i] - 2 * self.pt2 +  self.pt3[i]) - 2 * self.pt1[i] + 2 * self.pt2[i]

        return np.array(vectors)
        
    # UTILIZES FUNCTIONS IN THE BSPLINE LIBRARY TO CALCULATE THE POINT, TANGENT, AND BINORMAL VECTOR AT POINT T
    def get_point(self, t):
        pts_axis = np.array([self.pt1[i] * (1-t) ** 2 + 2 * (1-t) * t * self.pt2[i] + t ** 2 * self.pt3[i] for i in range(0, 3)])
        return pts_axis.transpose()  # return self.curve.evaluate(start=t, end=t) 

    def get_tangent(self, t):
        return np.array(operations.tangent(self.curve,t, normalize=True))  # returns a tuple of the origin pt and the vector components
    
    def get_binormal(self, t):
        # don't support operations.binormal(self.curve, t) anymore
        # return np.array(operations.binormal(self.curve, t))
    
        tangentVec = self.get_tangent(t)
        vecSecondDer = np.array([2 * (self.pt1[i] - 2.0 * self.pt2[i] + self.pt3[i]) for i in range(3)])

        binormalVec = np.cross(tangentVec, vecSecondDer)
        if np.isclose(np.linalg.norm(vecSecondDer), 0.0):
            for i in range(2):
                if not np.isclose(tangentVec[i], 0.0):
                    binormalVec[i] = -tangentVec[(i+1)%3]
                    binormalVec[(i+1)%3] = tangentVec[i]
                    binormalVec[(i+2)%3] = 0.0
                    break
        
        return binormalVec / np.linalg.norm(binormalVec)
       
    
    # Calculate the 4x4 matrix for a point t alont the curve
    # the matrix consists of the x vector, binormal vector, tangent vector, and center point
    def frenet_frame(self, t):
        centerPt = self.get_point(t)
        tangentVec = self.get_tangent(t)                # Normalized in the function (vector length == knot vector)
        binormalVec = self.get_binormal(t)              # vector length == knot vector 
        xVector = np.cross(tangentVec, binormalVec)     # vector length == knot vector
        
        # print("center pt", centerPt)
        # print("binormal", binormalVec)
        # print("tangent", tangentVec)
        # print("x vector", xVector)

        mat = np.identity(4)
        # need to take the second list in the vectors xVector, binormalVec, and tangentVec
        mat[0:3, 0] = xVector[1].transpose()
        mat[0:3, 1] = binormalVec[1].transpose()
        mat[0:3, 2] = tangentVec[1].transpose()
        mat[0:3, 3] = centerPt[0:3] 

        return mat
    
    # DESCRIPTION: calculate the radius of the branch from the beginning to the end
    # also accounts for if the branch is a junction point and there is a terminal bud 
    def calculate_radii(self):
        radii = np.linspace(self.startR, self.endR, self.along)

        # check if junction
        if self.startJunction:
            expR = self.startR * 0.25 * np.exp(np.linspace(0, -10.0, self.along))
            radii = radii + expR

        # check for terminal bud
        if self.terminalBud:
            startI = int(np.floor(self.startBud * self.along))
            totalI = self.along - startI
            budR = np.interp(np.linspace(0, 1, totalI), BranchGeometry._bud_shape[0, :], BranchGeometry._bud_shape[1, :])
            radii[startI:] *= budR
        
        return radii
    
    # calculates the cylinder shapes for each vertex point
    # used when creating the mesh
    def calc_cyl_vertices(self):
        pt = np.ones((4))
        radii = self.calculate_radii()

        for i, t in enumerate(np.linspace(0, 1.0, self.along)):  
            mat = self.frenet_frame(t)          # get the 4x4 matrix at the given point
            pt[0] = 0
            pt[1] = 0
            pt[2] = 0
            curvePt = mat @ pt                  # do matrix multiplication to determine the point
            for itheta, theta in enumerate(np.linspace(0, np.pi * 2.0, self.around, endpoint=False)):
                # get the radius at the specific point times the angle wrt sin or cos
                pt[0] = np.cos(theta) * radii[i]       
                pt[1] = np.sin(theta) * radii[i]  
                pt[2] = 0
                curvePt = mat @ pt

                # set the vertex location to the curve point on the branch given its position (t) and theta (itheta)
                self.vertexLocs[i, itheta, :] = curvePt[0:3].transpose()
    
    
    def make_cylinders(self, profiles):
        self.calc_cyl_vertices()

    def make_branch_segment(self, pt1, pt2, pt3, startR, endR, startJunction, terminalBud):
        self.set_ctrl_pts(pt1, pt2, pt3)
        self.set_branch_radius(startR, endR, startJunction, terminalBud)
        try:
            self.make_cylinders(BranchGeometry._profiles["tertiary"])
        except KeyError:
            self.make_cylinders(None)

    #######################################################################################################
    # DESCRIPTION: Determine where to place the buds based on the locations passed
    # INPUT:
    #      - locs: a list of tuples represting where to place the buds [(position along branch (0 to 1), angle (radians)), ...]
    #              - EX: [(0.25, 0), (0.5, 2.0 * np.pi/3), (0.75, 4.0 * np.pi/3)]
    # OUTPUT:
    #       - a list of points [(pt1, pt2, pt3), ...] representing the locations to place the buds
    ########################################################################################################
    def place_buds(self, locs):

        ts = np.linspace(0, 1, self.along) # get even spacing of points
        radii = self.calculate_radii()

        pt = np.ones((4))
        zeroPt = np.ones((4))
        zeroPt[0:3] = 0.0
        vec = np.ones((4))
        returnList = []

        for loc in locs:
            mat = self.frenet_frame(loc[0])     # get the matrix corresponding to the first point in the location
            r = np.interp(loc[0], ts, radii)    # interpolating the radius at loc[0] 
            curvePoint = mat @ zeroPt              # matrix multiplication

            pt[0] = np.cos(loc[1]) * r          
            pt[1] = np.sin(loc[1]) * r
            pt[2] = 0
            surfacePoint = mat @ pt

            vec[0] = np.cos(loc[1])
            vec[1] = np.sin(loc[1])
            vec[2] = 0
            rotateVec = np.cross(vec[0:3], np.array([0, 0, 1]))     
            rotateVec = rotateVec / np.linalg.norm(rotateVec)           # normalize the vector
            rotateBudMatrix = R.from_rotvec(self.budAngle * rotateVec)  # get the rotation matrix for the bud
            rotationMatrix = rotateBudMatrix.as_matrix()                # convert to a matrix

            vec[0:3] = rotationMatrix @ vec[0:3]
            curveVector = mat @ vec
            curveVector = curveVector * (self.budLength / np.linalg.norm(curveVector))
            endBudPoint = surfacePoint + curveVector
            midPoint = 0.7 * surfacePoint + 0.3 * curvePoint
            returnList.append((midPoint[0:3], surfacePoint[0:3], endBudPoint[0:3]))

        return returnList

    # FOR TESTING: taken from make_branch_geometry.py 
    # MODIFY to pass this mesh to the Mesh class to draw the resulting branch
    def write_mesh(self, fname):
        """Write out an obj file with the appropriate geometry
        @param fname - file name (should end in .obj"""
        with open(fname, "w") as fp:
            fp.write(f"# Branch\n")
            for it in range(0, self.along):
                for ir in range(0, self.around):
                    fp.write(f"v ") # writing the vertex
                    fp.write(" ".join(["{:.6}"] * 3).format(*self.vertexLocs[it, ir, :])) # writing the vertex location
                    fp.write(f"\n")
            for it in range(0, self.along - 1):
                i_curr = it * self.around + 1
                i_next = (it+1) * self.around + 1
                for ir in range(0, self.around):
                    ir_next = (ir + 1) % self.around
                    fp.write(f"f {i_curr + ir} {i_next + ir_next} {i_curr + ir_next} \n") # writing the faces for the function
                    fp.write(f"f {i_curr + ir} {i_next + ir} {i_next + ir_next} \n")

    # returns a list of faces
    # A helper function when passing the faces off to the meshes for collection
    def branch_faces(self):
        faces = []
        for it in range(0, self.along - 1):
            i_curr = it * self.around + 1
            i_next = (it + 1) * self.around + 1
            for ir in range(0, self.around):
                ir_next = (ir + 1) % self.around
                # need to subtract 1 so the list can capture the vertex lists or else it will be out of bounds
                faces.append([i_curr + ir - 1, i_next + ir_next - 1, i_curr + ir_next - 1])
                faces.append([i_curr + ir - 1, i_next + ir - 1, i_next + ir_next - 1])

        return faces
    
    # gets the initial ordering of the vertex list as shown in a .obj file
    def init_vertex_list(self):
        vertices = []
        for it in range(0, self.along):
            for ir in range(0, self.around):
                # print(self.vertexLocs[it, ir, :])
                vertices.append(self.vertexLocs[it, ir, :])
        return vertices

    # returns the correct order of vertices based on the faces values found earlier
    def branch_vertices(self):
        faces = self.branch_faces()
        initVert = self.init_vertex_list()
        vertices = []
        for face in faces:
            for vertex in face: # getting the vertex location
                vertFace = initVert[vertex]
                for v in vertFace:
                    vertices.append(round(v, 6))

        return np.array(vertices, dtype=ctypes.c_float)


    # uses existing branch values to add to create the branch
    def create_branch_points(self):
        # resets the control points
        self.set_ctrl_pts(pt1=[0.0, 0.0, 0.0], 
                          pt2=[self.branchLength/2, self.branchCurve, 0.0], 
                          pt3=[self.branchLength, 0.0, 0.0])
    
    
    # creates the branch based on the length, curve, and radius of the branch
    # returns the vertices and faces of the branch
    def create_branch(self):
        # branch = BranchGeometry(self.dir)
        self.create_branch_points()
        self.make_branch_segment(pt1 = self.pt1,
                                 pt2 = self.pt2,
                                 pt3 = self.pt3,
                                 startR = self.startR,
                                 endR = self.endR,
                                 startJunction = self.startJunction,
                                 terminalBud = self.terminalBud)
        self.faces = self.branch_faces()
        self.vertices = self.branch_vertices()
        # return branch
        # return faces, vertices

        

# main testing area  
if __name__ == '__main__':

    branch = BranchGeometry("data")
    # trunk radius start = 0.12, end = 0.07
    length = 0.3
    curveUp = 0.1
    curveSide = 0.05

    startRad = 0.03
    endRad = 0.01

    tertiaryPt1 = [0.0, 0.0, 0.0] # where the bottom point aligns. TO CHANGE when know point on tree to attach
    tertiaryPt2 = [length/2, curveUp, curveSide]
    tertiaryPt3 = [length, 0.0, 0.0]

    branch.make_branch_segment(pt1=tertiaryPt1, 
                               pt2=tertiaryPt2, 
                               pt3=tertiaryPt3, 
                               startR=startRad, 
                               endR=endRad, 
                               startJunction=True, 
                               terminalBud=True)

    # branch.write_mesh(f"tree_files/testSplineBranch.obj")

    face = branch.faces
    print(face[:10])
    print(branch.vertices[:10])
