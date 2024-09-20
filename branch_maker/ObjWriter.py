"""
Has 2 classes reasonable to use outside of this file: ObjWriter and EndConnectionMode\n
ObjWriter handles writing to an obj file. It can write comments, vertices, texture coordinates, faces, cubes, rotation indicators (useful for debug purposes), and most importantly Tubes\n
EndConnectionMode is just a Enum to configure settings for writing tubes\n
This file also has the class Vertex which is meant for internal use to track indices of verts, cord, and norms. \n

>================| Key Words |================<\n
vert -> vertex -> "v" in obj\n
cord -> vertex texture coordinate -> "vt" in obj\n
norm -> vertex normal -> "vn" in obj\n
face -> face -> "f" in obj\n
comment -> comment -> "#" in obj
"""

import copy
from enum import Enum
from io import TextIOWrapper
from typing import List, Tuple, Union
import math
from Vec3 import Vec3
from Ring import Ring
from Tube import Tube 
from scipy.spatial.transform import Rotation


def round_float(value: float, rounding: int) -> str:
    """Format the float with the specified number of decimal places"""
    formatted_value = f"{value:.{rounding}f}"
    # Remove trailing zeros and possible trailing decimal point
    result = formatted_value.rstrip('0').rstrip('.') if '.' in formatted_value else formatted_value
    if result == "-0": return "0"
    return result


class EndConnectionMode(Enum):
    """How display a tube.\n
    >>  NONE - not caps,\n
    >>  CAPS ngons on both ends,\n
    >>  CAP_FRONT and CAP_BACK do one ngon an one hole,\n
    >>  CONNECTED connects the two sides together so like a donut, !NOT WORKING RN!\n
    >>  DISCS does not do the body at all just does ngon for every ring\n
    >>  CUBES does a cube at midpoint of every ring scaled to radius/2\n
    note: if triangulation is on then ngons are also tris"""
    NONE = 0
    CAPS = 1
    CAP_FRONT = 2
    CAP_BACK = 3
    CONNECTED = 4
    DISCS = 5
    CUBES = 6

class Vertex:
    """Tracks indices of verts, cords, and norms\n
    aka vertex indices, vertex texture coordinate indices, and vertex normal indices"""
    vertIndex: int
    cordIndex: int
    normIndex: int

    def __init__(self, vertIndex:int, cordIndex:int, normIndex: int):
        self.vertIndex = vertIndex
        self.cordIndex = cordIndex
        self.normIndex = normIndex

    def __add__(self, other:Union['Vertex', int, Tuple[int, int, int]]) -> 'Vertex':
        if isinstance(other, Vertex):
            return Vertex(self.vertIndex + other.vertIndex, self.cordIndex + other.cordIndex, self.normIndex + other.normIndex)
        elif isinstance(other, int):
            return Vertex(self.vertIndex + other, self.cordIndex + other, self.normIndex + other)
        else:
            return Vertex(self.vertIndex + other[0], self.cordIndex + other[1], self.normIndex + other[2])
        
    @staticmethod
    def fromSingleArray(input: Union[List[int],range]) -> List['Vertex']:
        return Vertex.fromArrays(input,input, input)
    
    @staticmethod
    def fromArrays(vertIndexArray: Union[List[int],range], cordIndexArray: Union[List[int],range], normIndexArray: Union[List[int],range,int]) -> List['Vertex']:
        result = []
        for i in range(len(vertIndexArray)):
            normIndexValue = normIndexArray if isinstance(normIndexArray, int) else normIndexArray[i]
            result.append(Vertex(vertIndexArray[i],cordIndexArray[i],normIndexValue))
        return result
    
    @staticmethod
    def fromTupleArray(input: list[Tuple[int,int,int]]) -> List['Vertex']:
        result = []
        for i in range(len(input)):
            result.append(Vertex(input[i][0],input[i][1],input[i][2]))
        return result
    
    def __repr__(self):
        return f"Vertex<{self.vertIndex} {self.cordIndex} {self.normIndex}>"

class ObjWriter:
    """
    handles writing to an obj file.
    methods that start with `add` dont update `offset` and directly modify the file.
    methods that start with `write` must update `offset` and call various other `add` and/or `write` methods.
    methods that start with `_` are for internal use.
    all felids are private, and ones that start with `_` are extra private only to be used by `add` methods.

    A note on offsets:
    In objs, the index of a vert, cord, and norm all need to be tracked for when creating faces.
    These numbers are independent of each other.
    To know what the index of a new vert/cord/norm is, you need to know the index of the latest vert/cord/norm. This is called the offset.
    All three are often tracked together in an instance of Vertex.
    An `add` method does not increment these offsets, while a `write` method does.
    Different `write` methods might increment them at different times as long as it offset is settled up by the end.
    
    However, this matter is complicated by caching.
    cords and norms are 'cached', meaning that:
    if a cord/norm with given values (after rounding) has already been saved to the obj file, it will not add another identical one and will instead reuse the previous cord/norm.
    This process is handled completely by the `add` methods and will not need be worried about within the `write` methods - `write` functions as if no caching is occurring and the `add` methods do the logic beneath the surface to ensure everything is correct.

    How it works is that:
    a secondary index tracking (_real_cord_offset/_real_norm_offset) is used to know the 'real' offset written to the file.
    a dictionary maps the 'fake' offsets (used in the `write` methods and saved in `offset`) to the 'real' offsets (written into the file).
    Then when a new cord/norm is added it will check if its new and if it is then write to the file and add it to the dictionary (real_cord_cache/real_norm_cache).
    When a cord/norm is added that can be reused, it writes to the dictionaries that this 'fake index' is mapped to the given 'real index'.
    The ints _real_cord_offset/_real_norm_offset track what has been written to the file.
    The ints _fake_cord_offset/_fake_norm_offset track how many times _addCord/_addNorm have been called.
    The dictionaries real_cord_cache/real_norm_cache map both uv/position and fake indices to real indices.
    
    Why cache?
    Reduces the outputted file size some.
    
    Why not verts or faces?
    two objects intersecting each other that happen to vertices on top of each other would become connected. This does not matter in regards to static models obj, but does if further modified because the objects might become connected.
    Also an event of caching either a vert or a face would be very rare - in comparison to caching a cord or a norm"""

    fp: TextIOWrapper
    offset: Vertex
    """The offsets of different indices before caching that should be up to date by the end of every write method"""
    triangulate: bool
    """force everything to be triangles?"""
    real_cord_cache: dict[Union[str,int], int]
    """str is the UV position (rounded values with space between as a string) which maps to a real index
    int is a fake index which maps to a real index"""
    _real_cord_offset: int
    """Used for the caching"""
    _fake_cord_offset: int
    """Used for the caching"""
    real_norm_cache: dict[Union[str,int], int]
    """str is the normal (rounded values with space between as a string) which maps to a real index
    int is a fake index which maps to a real index"""
    _real_norm_offset: int
    """Used for the caching"""
    _fake_norm_offset: int
    """Used for the caching"""
    vert_rounding: int
    """how many places the numbers of each type should be rounded to"""
    cord_rounding: int
    """how many places the numbers of each type should be rounded to"""
    norm_rounding: int
    """how many places the numbers of each type should be rounded to"""

    def __init__(self, fp: TextIOWrapper, title: str = "", triangulate: bool = False, vert_rounding: int = 4, cord_rounding: int = 3, norm_rounding: int = 2) -> None:
        """A recommended way to start this is:\n
        with open("output.obj", "w") as fp:\n
            obj = ObjWriter(fp)\n
        vert_rounding, cord_rounding, norm_rounding are how many places the numbers of each type should be rounded to. Defaults are designed to be just above the minimum for the application of small thin tree branches.
        if triangulate is off (default), then resulting meshes can (and most likely will) contain quads and ngons. If enabled, everything will be converted into tris.
        title is optional and will just add a comment at the top of the obj with it
        """
        self.fp = fp
        self.offset = Vertex(1,1,1)
        self.triangulate = triangulate
        # self._smooth_shading = False
        self.real_cord_cache = {}
        self._real_cord_offset = 1
        self._fake_cord_offset = 1
        self.real_norm_cache = {}
        self._real_norm_offset = 1
        self._fake_norm_offset = 1
        self.vert_rounding = vert_rounding
        self.cord_rounding = cord_rounding
        self.norm_rounding = norm_rounding
        if title != "":
            self._addComment(title)


    def _addVert(self, pos: Vec3) -> None:
        """Adds a vert aka vertex. Does not increment self.offset.vertIndex"""
        self.fp.write(f"v {round_float(pos.x,self.vert_rounding)} {round_float(pos.y,self.vert_rounding)} {round_float(pos.z,self.vert_rounding)}\n")


    def _addCord(self, x: float, y: float) -> None:
        """Adds a cord aka vertex texture coordinate. Does not increment self.offset.cordIndex"""
        result = f"{round_float(x,self.cord_rounding)} {round_float(y,self.cord_rounding)}"
        if result in self.real_cord_cache:
            real_index = self.real_cord_cache[result]
            self.real_cord_cache[self._fake_cord_offset] = real_index
        else:
            self.real_cord_cache[result] = self._real_cord_offset
            self.real_cord_cache[self._fake_cord_offset] = self._real_cord_offset
            self._real_cord_offset += 1
            self.fp.write(f"vt {result}\n")

        self._fake_cord_offset += 1


    def _addNorm(self, vec: Vec3) -> None:
        """Adds a norm aka vertex normal. Does not increment self.offset.normIndex"""
        result = f"{round_float(vec.x,self.norm_rounding)} {round_float(vec.y,self.norm_rounding)} {round_float(vec.z,self.norm_rounding)}"
        if result in self.real_norm_cache:
            real_index = self.real_norm_cache[result]
            self.real_norm_cache[self._fake_norm_offset] = real_index
        else:
            self.real_norm_cache[result] = self._real_norm_offset
            self.real_norm_cache[self._fake_norm_offset] = self._real_norm_offset
            self._real_norm_offset += 1
            self.fp.write(f"vn {result}\n")

        self._fake_norm_offset += 1


    def _addFace(self, vertex_indices: Union[List[Vertex],range], offset: Union[bool,Tuple[bool,bool,bool]] = True) -> None:
        """"
        Adds a face based on a list or range of vertex indices.
        if its a range then Vertex.fromSingleArray is used. In this case it is recommended to use an offset of True.

        offset controls wether the indices are absolute (false) or relative to self.offset (true).
        `True` is the same as `(True, True, True)`.
        The first corresponds the the vert, second corresponds to cord, third corresponds to norm.

        If `triangulate` is false it will add one face.

        If `triangulate` is true it will add len(vertex_indices) - 2 faces with 3 verts each.
        The first vert and ever adjacent pair of verts (excluding the first) will have a triangle face.

        The cord and norm indices placed in file are 'real' indices not the 'fake' indices used by self.offset.
        They are translated from 'fake' to real with the help of real_cord_cache/real_norm_cache.
        See the note on the class for more information on cord/norm caching.
        """
        real_vertex_indices:List[Vertex]
        if isinstance(vertex_indices,range):
            real_vertex_indices = Vertex.fromSingleArray(vertex_indices)
        else:
            real_vertex_indices = vertex_indices

        offsetVerts: bool
        offsetCords: bool
        offsetNorms: bool
        if isinstance(offset, bool):
            offsetVerts = offset
            offsetCords = offset
            offsetNorms = offset
        else:
            offsetVerts = offset[0]
            offsetCords = offset[1]
            offsetNorms = offset[2]

        if self.triangulate:
            for i in range(len(real_vertex_indices) - 2):
                A = copy.copy(real_vertex_indices[0])
                B = copy.copy(real_vertex_indices[i+1])
                C = copy.copy(real_vertex_indices[i+2])

                if offsetVerts:
                    A.vertIndex += self.offset.vertIndex
                    B.vertIndex += self.offset.vertIndex
                    C.vertIndex += self.offset.vertIndex
                if offsetCords:
                    A.cordIndex += self.offset.cordIndex
                    B.cordIndex += self.offset.cordIndex
                    C.cordIndex += self.offset.cordIndex
                if offsetNorms:
                    A.normIndex += self.offset.normIndex
                    B.normIndex += self.offset.normIndex
                    C.normIndex += self.offset.normIndex

                real_A_cord = self.real_cord_cache[A.cordIndex]
                real_B_cord = self.real_cord_cache[B.cordIndex]
                real_C_cord = self.real_cord_cache[C.cordIndex]
                real_A_norm = self.real_norm_cache[A.normIndex]
                real_B_norm = self.real_norm_cache[B.normIndex]
                real_C_norm = self.real_norm_cache[C.normIndex]

                self.fp.write(f"f {A.vertIndex}/{real_A_cord}/{real_A_norm} {B.vertIndex}/{real_B_cord}/{real_B_norm} {C.vertIndex}/{real_C_cord}/{real_C_norm}\n")
        else:
            face = "f"
            for vert in real_vertex_indices:
                index = copy.copy(vert)

                if offsetVerts:
                    index.vertIndex += self.offset.vertIndex
                if offsetCords:
                    index.cordIndex += self.offset.cordIndex
                if offsetNorms:
                    index.normIndex += self.offset.normIndex

                real_cord = self.real_cord_cache[index.cordIndex]
                real_norm = self.real_norm_cache[index.normIndex]

                face += f" {index.vertIndex}/{real_cord}/{real_norm}"
            self.fp.write(face + "\n")


    def _addComment(self, comment: str) -> None:
        self.fp.write(f"# {comment}\n")


    def writeComment(self, comment: str) -> None:
        """its just an atlas for _addComment"""
        self._addComment(comment)


    def _writeCircleVerts(self, number: int, radius: float = 0.5) -> List[int]:
        """add vertex cords
        returns list of indices of those
        """
        result = []
        for vertNum in range(number):
            theta = vertNum/number*math.tau
            self._addCord(math.cos(theta)*radius+0.5, math.sin(theta)*radius+0.5)
            result.append(self.offset.cordIndex)
            self.offset.cordIndex+=1
        return result

    def _addASingleFlatNorm(self, verts: List[Vec3]) -> None:
        """Finds the normal of a plane best fitting some point and uses that as a normal"""
        normal = Vec3.fromPlaneFitting(verts)
        self._addNorm(normal)


    def writeSingleFace(self, verts: List[Vec3], at: Vec3 = Vec3.zero()) -> None:
        """Create vertices and connects them as a face. The texture coordinates assume that verts are a circle. All the norms face the best assumed direction"""
        for vert in verts:
                self._addVert(vert + at)
        self._addASingleFlatNorm(verts)
        vertIndices = range(len(verts))
        cordIndices = self._writeCircleVerts(len(verts))
        self._addFace(Vertex.fromArrays(vertIndices, cordIndices, 0), (True, False, True))
        self.offset.vertIndex += len(verts)
        self.offset.normIndex += 1


    def writeDirectionIndicator(self, size: float, rotation: Rotation, at: Vec3 = Vec3.zero()) -> None:
        """Is a disc with a triangle - the side of the disc with the triangle is front. The direction of the triangle is up"""
        radius = size / 2
        self.writeSingleFace(Ring(Vec3.zero(), radius, rotation).convert_to_verts(6), at)
        self.writeSingleFace([
            Vec3.zero(),
            Vec3.Y() * rotation * radius,
            Vec3(0.5,1,0) * rotation * radius
        ], at)

    def writeCube(self, sideLength: float, at: Vec3 = Vec3.zero()) -> None:
        # self.writeShadeFlat()
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    self._addVert(Vec3(x-0.5, y-0.5, z-0.5)*sideLength + at)
        
        self._addCord(0,0)
        self._addCord(0,1)
        self._addCord(1,1)
        self._addCord(1,0)
        self._addNorm(-Vec3.X())
        self._addNorm(Vec3.X())
        self._addNorm(-Vec3.Y())
        self._addNorm(Vec3.Z())
        self._addNorm(Vec3.Y())
        self._addNorm(-Vec3.Z())
        
        self._addFace(Vertex.fromTupleArray([(0,0,0),(1,1,0),(3,2,0),(2,3,0)]), True)
        self._addFace(Vertex.fromTupleArray([(6,0,1),(7,1,1),(5,2,1),(4,3,1)]), True)
        self._addFace(Vertex.fromTupleArray([(4,0,2),(5,1,2),(1,2,2),(0,3,2)]), True)
        self._addFace(Vertex.fromTupleArray([(5,0,3),(7,1,3),(3,2,3),(1,3,3)]), True)
        self._addFace(Vertex.fromTupleArray([(7,0,4),(6,1,4),(2,2,4),(3,3,4)]), True)
        self._addFace(Vertex.fromTupleArray([(6,0,5),(4,1,5),(0,2,5),(2,3,5)]), True)

        self.offset += (8,4,6)


    def _writeDiscTube(self, tubeVerts: List[List[Vec3]], at: Vec3 = Vec3.zero()) -> None:
        for ring in tubeVerts:
            self.writeSingleFace(ring, at)


    def _writeCubeTube(self, tube: 'Tube', at: Vec3 = Vec3.zero()) -> None:
        for ring in tube:
            self.writeCube(ring.radius/2,ring.center+at)


    def writeTube(self, tube: 'Tube', vertsAround: int, endConnectionMode: EndConnectionMode = EndConnectionMode.NONE, at: Vec3 = Vec3.zero()) -> None:
        """Writes a tube to the obj.
        
        vertAround is the how many verts go around each ring.

        if the result looks ugly try using:
        
        tube.apply_parallel_transport()

        tube = tube.subdivide().subdivide()

        tube.simplify()

        then passing it in
        """

        if endConnectionMode == EndConnectionMode.CONNECTED:
            raise NotImplementedError("EndConnectionMode.CONNECTED is currently broken :( pls dont use")

        if len(tube) == 0:
            return

        if endConnectionMode == EndConnectionMode.CUBES:
            self._writeCubeTube(tube, at)
            return 

        # reformat tube
        tubeVerts: List[List[Vec3]] = []
        for ring in tube:
            tubeVerts.append(Ring(ring.center, ring.radius, ring.rotation).convert_to_verts(vertsAround))
        
        # discs is created so differently that its code has been moved to a different method
        if endConnectionMode is EndConnectionMode.DISCS:
            self._writeDiscTube(tubeVerts, at)
            return 
        
        if len(tubeVerts) == 1:
            if endConnectionMode != EndConnectionMode.NONE:
                self.writeSingleFace(tubeVerts[0], at)
            return

        # define some constants
        rings = len(tubeVerts)
        verts_per_ring = len(tubeVerts[0])
        cords_per_ring = verts_per_ring + 1 # because the first and last face share the same vert but use different cords, there is 1 more cord than verts per ring
        totalNumberOfVerts = rings * verts_per_ring
        totalNumberOfCords = rings * cords_per_ring
        # the total number of norms is totalNumberOfVerts or totalNumberOfVerts+1 or totalNumberOfVerts+2 depending on the endConnectionMode

        # add verts
        for ring_index, ring in enumerate(tubeVerts):
            for vert_index, vert in enumerate(ring):
                self._addVert(vert + at)
                self._addCord(tube.lengths[ring_index]/tube.total_length(), vert_index/float(verts_per_ring))
                # simply gets the verts positions adjacent to the current vert
                # verts ahead and behind can be none, but the verts lefts and right wrap around
                # these positions are used to determine the normal
                normal = vert.get_normal_from_around(
                    ahead = tubeVerts[ring_index+1][vert_index] if ring_index + 1 < rings else None,
                    behind = tubeVerts[ring_index-1][vert_index] if ring_index - 1 >= 0 else None,
                    left = ring[(vert_index+1)%verts_per_ring],
                    right = ring[(vert_index-1)%verts_per_ring]
                )
                self._addNorm(normal)
            # one extra texture coordinate per ring than normals or verts
            # |--|--|--|--?
            # if | is vert and -- is face then ? wraps around to the first vert, but texture cords are 2d and cant wrap so it extends one farther
            self._addCord(tube.lengths[ring_index]/tube.total_length(), 1)
            

        # add normal faces (not caps/discs)
        # self.writeShadeSmooth()
        ringsToAddFacesTo: int
        if endConnectionMode is EndConnectionMode.CONNECTED:
            ringsToAddFacesTo = rings
        else:
            ringsToAddFacesTo = rings - 1
        
        for ring_index in range(ringsToAddFacesTo):
            # defining some constants
            # norms are identical logic of verts - except they use a different offset
            first_vert_of_ring = ring_index * verts_per_ring + self.offset.vertIndex
            first_vert_of_next_ring = ((ring_index + 1) * verts_per_ring) % (totalNumberOfVerts) + self.offset.vertIndex
            first_cord_of_ring = ring_index * cords_per_ring + self.offset.cordIndex
            first_cord_of_next_ring = (ring_index + 1) * cords_per_ring  + self.offset.cordIndex
            first_norm_of_ring = ring_index * verts_per_ring + self.offset.normIndex
            first_norm_of_next_ring = ((ring_index + 1) * verts_per_ring) % (totalNumberOfVerts) + self.offset.normIndex
            for vertex_index in range(verts_per_ring):
                # defining some constants
                # vA means - v (vert) and A (first vert of the quad to be added)
                # cC means - c (cord) and C (third vert of the quad to be added)
                next_vert_index = (vertex_index + 1) % verts_per_ring
                vA = first_vert_of_ring + vertex_index
                vB = first_vert_of_ring + next_vert_index
                vC = first_vert_of_next_ring + next_vert_index
                vD = first_vert_of_next_ring + vertex_index
                next_cord_index = (vertex_index + 1)
                cA = first_cord_of_ring + vertex_index
                cB = first_cord_of_ring + next_cord_index
                cC = first_cord_of_next_ring + next_cord_index
                cD = first_cord_of_next_ring + vertex_index
                nA = first_norm_of_ring + vertex_index
                nB = first_norm_of_ring + next_vert_index
                nC = first_norm_of_next_ring + next_vert_index
                nD = first_norm_of_next_ring + vertex_index
                self._addFace([Vertex(vA,cA,nA), Vertex(vB,cB,nB), Vertex(vC,cC,nC), Vertex(vD,cD,nD)], False) # False means these are absolute indices


        # verts' offset is handled after potential adding caps (because verts are reused, but cords and norms are not for caps)
        self.offset.cordIndex += totalNumberOfCords
        self.offset.normIndex += totalNumberOfVerts

        if endConnectionMode is EndConnectionMode.CAPS or endConnectionMode is EndConnectionMode.CAP_FRONT:
            # self.writeShadeFlat()
            # add front caps
            vertIndexArray = range(self.offset.vertIndex, self.offset.vertIndex + verts_per_ring)
            cordIndexArray = self._writeCircleVerts(verts_per_ring, 1/math.tau) # 1/math.tau so that the texture scaling of the caps are more similar to the that of the front of the tube (the ratio of radius to circumference is 1/tau)
            self._addASingleFlatNorm(tubeVerts[0])
            # absolute vert indices, absolute cord indices, absolute cord indices, relative norm indices
            self._addFace(Vertex.fromArrays(vertIndexArray,cordIndexArray,0)[::-1], (False, False, True)) # idk why but it was backfacing here and nowhere else. [::-1] makes up for that.
            self.offset.normIndex += 1

        if endConnectionMode is EndConnectionMode.CAPS or endConnectionMode is EndConnectionMode.CAP_BACK:
            # self.writeShadeFlat()
            # add back caps
            firstVertOfLastRing = self.offset.vertIndex + totalNumberOfVerts - verts_per_ring
            vertIndexArray = range(firstVertOfLastRing, firstVertOfLastRing + verts_per_ring)
            cordIndexArray = self._writeCircleVerts(verts_per_ring, 1/math.tau) # 1/math.tau so that the texture scaling of the caps are more similar to the that of the front of the tube (the ratio of radius to circumference is 1/tau)
            self._addASingleFlatNorm(tubeVerts[-1])
            # absolute vert indices, absolute cord indices, absolute cord indices, relative norm indices
            self._addFace(Vertex.fromArrays(vertIndexArray, cordIndexArray,0), (False, False, True))
            self.offset.normIndex += 1
            # self._addFace(range(firstVertOfLastRing, firstVertOfLastRing + verts_per_ring), False)

        self.offset.vertIndex += totalNumberOfVerts