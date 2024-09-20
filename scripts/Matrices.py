import numpy as np
import math


class Matrices:
    def __init__(self, type):
        self.matrix = np.zeros((4, 4))
        self.type = type

    # Creates a projection matrix based on:
    #   - fovy: The field of view given in degrees from the center line
    #   - aspect: The aspect ratio (often width / height) of the screen
    #   - near: the near plane of the window (i.e., for depth of z)
    #   - far: the far plane of the window (i.e., for depth of z)
    def projection(self, fovy, aspect, near, far):
        s = 1.0 / math.tan(math.radians(fovy)/2.0) 
        sx, sy = s / aspect, s

        zz = (far + near) / (near - far)
        zw = 2 * far * near / (near - far)

        return np.matrix([[sx, 0, 0, 0],
                          [0, sy, 0, 0],
                          [0, 0, zz, zw],
                          [0, 0, -1, 0]])

# return normalVector of the plane
def normalVector(v1, v2, v3):
    return np.cross(v2 - v1, v3 - v1) 
 
###############################
# Algorithm inspired by: https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
################################
def rayCast(origin, rayDirection, plane):
    # take a 3D direction and calculate if it intersects the plane
    normal = normalVector(plane[0], plane[1], plane[2])
    # area = np.linalg.norm(normal)
    # print(normal)

    denominator = np.dot(rayDirection, normal)    
    if denominator <= 1e-3: # close to 0 emaning it is almost parallel
        return None # no intersect due to plane and ray being parallel
    
    # dist = origin - plane[0] # change from A to center point of the plane
    dist = np.dot(-normal, plane[0])
    numerator = -(np.dot(normal, origin) + dist)
    # numerator = np.dot(origin, normal) + plane[0] # to the distance of pt1 on the plane
    t = numerator / denominator
    if t < 0:
        return None # triangle is behind the ray
    
    pt = origin + t * rayDirection # possible intercept point
    
    # DETERMINING IF THE RAY INTERXECTS USING BARYCENTRIC COORDINATES
    edgeBA = plane[1] - plane[0]
    pEdgeA = pt - plane[0] # vector between possible intercept point and pt A of the triangle
    perp = np.cross(edgeBA, pEdgeA) # vector perpendicular to triangle's plane
    u = np.dot(normal, perp)
    if u < 0:
        return None

    edgeCB = plane[2] - plane[1]
    pEdgeB = pt - plane[1]
    perp = np.cross(edgeCB, pEdgeB)
    v = np.dot(normal, perp)
    if v < 0:
        return None
    
    edgeAC = plane[0] - plane[2]
    pEdgeC = pt - plane[2]
    perp = np.cross(edgeAC, pEdgeC)
    w = np.dot(normal, perp)
    if w < 0:
        return None

    return pt


def interception(origin, rayDirection, vertices):
    # Need to loop through the list of all vertex positions
    # grab out each face, and pass that to the rayCast algorithm
    # append intercepts to a list of values 
    intercepts = []
    for i in range(int(vertices.size / 9)):
        face = vertices[3*i:3*i+3]
        # print(f"Face from index {3*i} to {3*i+3} is {face}")

        pt = rayCast(origin, rayDirection, face)
        if pt is not None:
            intercepts.append(pt)
            print(f"Intercept at {pt}")
    
    return intercepts

    
if __name__ == '__main__':
    # plane
    pt1 = [2, 0, -2] # -2
    pt2 = [0, -2, -2] # -2
    pt3 = [0, 2, -2] 
    pt4 = [-2, 0, -2]
    pt5 = [0, 0, -1]

    # Points for a pyramid
    # Need to combine to make the faces

    vertices = np.array([pt5, pt1, pt3, 
                         pt5, pt3, pt4,
                         pt5, pt1, pt2, 
                         pt5, pt4, pt2, 
                         pt2, pt3, pt4, 
                         pt1, pt2, pt3], dtype=np.float32) 

    # vertices = np.array([0, 0, -1, 2, 0, -2, 0, 2, -2, 
    #                      0, 0, -1, 0, 2, -2, -2, 0, -2,
    #                      0, 0, -1, 2, 0, -2, 0, -2, -2, 
    #                      0, 0, -1, -2, 0, -2, 0, -2, -2, 
    #                      0, -2, -2, 0, 2, -2, -2, 0, -2, 
    #                      2, 0, -2, 0, -2, -2, 0, 2, -2], dtype=np.float32)   
    

    faces = [[pt5, pt1, pt3],[pt5, pt3, pt4]]
    pt1 = np.array([3, -2])
    pt2 = np.array([-1, 1])
    maxX = 3
    minX = -1
    maxY = 1
    minY = -2
    # maxX = np.max(pt1[0], pt2[0])
    # maxY = np.max(pt1[1], pt2[1])
    # minX = np.min(pt1[0], pt2[0])
    # minY = np.min(pt1[1], pt2[1])
    intercepts = []
    for face in faces:
        # Need to call np.where
        faceIntercept = False
        for vertex in face:
            print(f"{vertex[0]}, {vertex[1]}")

            # check one of the vertices is in the region bounded by the line
            result = np.where((vertex[0] >= minX) and (vertex[0] <= maxX) and (vertex[1] >= minY) and (vertex[1] <= maxY))

            if len(result) > 0:
                faceIntercept = True
        
        # check if the line is possibly inside a face as well:
        # mainly check if within the bounding box of the face
        # Need to check if one or both vertices are inside the 
        triangleMinX = np.min([face[0][0], face[1][0], face[2][0]])
        triangleMaxX = np.max([face[0][0], face[1][0], face[2][0]])
        triangleMinY = np.min([face[0][1], face[1][1], face[2][1]])
        triangleMaxY = np.max([face[0][1], face[1][1], face[2][1]])

        pt1Inside = (pt1[0] >= triangleMinX) and (pt1[0] <= triangleMaxX) and (pt1[1] >= triangleMinY) and (pt1[1] <= triangleMaxY)
        pt2Inside = (pt2[0] >= triangleMinX) and (pt2[0] <= triangleMaxX) and (pt2[1] >= triangleMinY) and (pt2[1] <= triangleMaxY)

        if pt1Inside or pt2Inside:
            faceIntercept = True
        
        if faceIntercept:
            intercepts.append(face)

    print(intercepts)
    # Need to test if the line could theoretically cross a face within the bounding box of x1, y1 and x2, y2
    # want to look at their faces





    # print(vertices)
    # plane = np.array([pt1, pt2, pt3])

    # rayDirection = np.array([0, -1, -1])
    origin = np.array([0, 0, 0])
    line = np.array([[1, -1, 0], [-1, 1, 0]])
    rayDirection = (line[0] + line[1]) / 2
    rayDirection[2] = -1
    
    # print(rayDirection) 
    

    # intercepts = interception(origin, rayDirection, vertices)
    # if len(intercepts) == 0:
    #     print("No intercept found")
    # print(intercepts)
    # print("Intercept", rayCast(origin, rayDirection, plane))





    


    

    
