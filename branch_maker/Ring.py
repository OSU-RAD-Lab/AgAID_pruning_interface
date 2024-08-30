"""
Ring is a circle in 3d space with a radius and optionally a rotation
"""

import copy
from typing import Iterable, Iterator, List, Optional, Tuple
import math
from Vec3 import Vec3
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from Interpolate import *


class Ring:
    """A circle in 3d space with a radius and optionally a rotation"""
    center: Vec3
    radius: float
    _rotation: Optional[Rotation]
    """use .rotation to get the rotation to not have to deal with none states"""


    def __init__(self, center: Vec3 = Vec3.zero(), radius: float = 0.1, rotation: Optional[Rotation] = None):
        self.center = center
        self.radius = radius
        self._rotation = rotation


    def is_rotation_set(self) -> bool:
        return self._rotation is None


    @property
    def rotation(self) -> Rotation:
        if self._rotation is None:
            return Rotation.identity()
        else:
            return self._rotation
        

    @property
    def rotation_or_none(self) -> Optional[Rotation]:
        return self._rotation
        

    @rotation.setter
    def rotation(self, rotation: Rotation) -> None:
        self._rotation = rotation


    def convert_to_verts(self, resolution: int) -> List[Vec3]:
        """the most important method in this class probably"""
        returnList = []

        for vertIndex in range(0, resolution):
            radians = vertIndex / resolution * math.tau
            vector = Vec3(0, math.cos(radians), math.sin(radians)) * self.radius
            vertexLocation = self.center + vector.transform(self.rotation)
            returnList.append(vertexLocation)
        
        return returnList
    

    def __repr__(self) -> str:
        return f"Ring<Center:{self.center} Radius:{self.radius} Rotation:{self.rotation.as_euler('xyz')}>"
    

    def lerp(self, other: 'Ring', t: float) -> 'Ring':
        """Have you ever had two rings which you like, but neither are perfect? well today is your luck day because you can linearly interpolate between two rings with this method!! (that was a lie lmao the rotation is actually SLERPed instead of LERPed.)"""
        if self is other:
            return copy.copy(self)
        center: Vec3 = self.center.lerp(other.center, t)
        radius: float =  lerp(self.radius, other.radius, t)
        rotation: Rotation = Slerp([0,1], Rotation.concatenate([self.rotation, other.rotation]))(t)
        return Ring(center, radius, rotation)


    def toDict(self) -> dict:
        """make a dictionary describing this ring and stores the rotation as a quaternion."""
        rotation_as_quat = self.rotation.as_quat()
        return {
            "position": self.center.toDict(),
            "radius": self.radius,
            "rotation": {
                "w": rotation_as_quat[0],
                "x": rotation_as_quat[1],
                "y": rotation_as_quat[2],
                "z": rotation_as_quat[3]
            }
        }