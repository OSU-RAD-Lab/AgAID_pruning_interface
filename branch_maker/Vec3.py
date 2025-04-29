"""
A utility to represent 3d coordinates and directions\n
can be used as a 3 sized list\n
Vec3(1,0,0) * Rotation returns a rotated Vec3\n
Vec3(1,0,0).np gives a numpy list\n
Vec3(*np.array([1,0,0])) to convert from np array to Vec3\n
can be added, subtracted multiplied and divided as you otherwise expect\n
see methods for other things it can do
"""

from typing import Iterator, List, Optional, Union
import math
from scipy.spatial.transform import Rotation
import numpy as np

class Vec3:
    """
    A utility to represent 3d coordinates and directions\n
    can be used as a 3 sized list\n
    Vec3(1,0,0) * Rotation returns a rotated Vec3\n
    Vec3(1,0,0).np gives a numpy list\n
    Vec3(*np.array([1,0,0])) to convert from np array to Vec3\n
    can be added, subtracted multiplied and divided as you otherwise expect\n
    see methods for other things it can do
    """

    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __neg__(self) -> 'Vec3':
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, other: Union['Vec3', float, int, Rotation]) -> 'Vec3':
        """Multiplying by vector does pairwise multiplication.\n
        Multiplying by a float/int multiplies all values.\n
        Multiplying by a rotation transforms the vector by the rotation."""
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Rotation):
            return self.transform(other)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Vec3' and '{}'".format(type(other).__name__))

    def __rmul__(self, other: Union['Vec3', float, int, Rotation]) -> 'Vec3':
        """Multiplying by vector does pairwise multiplication.\n
        Multiplying by a float/int multiplies all values.\n
        Multiplying by a rotation transforms the vector by the rotation."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union['Vec3', float, int]) -> 'Vec3':
        """dividing by a vector returns does pair wise division\n
        dividing by a float/int divides all of the vector by those values"""
        if isinstance(other, Vec3):
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed")
            return Vec3(self.x / other, self.y / other, self.z / other)
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Vec3' and '{}'".format(type(other).__name__))
    
    def __rtruediv__(self, other: Union['Vec3', float, int]) -> 'Vec3':
        """dividing by a vector returns does pair wise division\n
        dividing by a float/int divides all of the vector by those values"""
        if isinstance(other, Vec3):
            return Vec3(other.x / self.x, other.y / self.y, other.z / self.z)
        if isinstance(other, (int, float)):
            return Vec3(other / self.x, other / self.y, other / self.z)
        else:
            raise TypeError("Unsupported operand type(s) for /: '{}' and 'Vec3'".format(type(other).__name__))

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError(f"Index {index} out of range Vec3")
        # match index:
        #     case 0:
        #         return self.x
        #     case 1:
        #         return self.y
        #     case 2:
        #         return self.z
        #     case _:
        #         raise IndexError(f"Index {index} out of range of Vec3")
        
    def __len__(self) -> int:
        return 3
    
    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __contains__(self, item: float) -> bool:
        return item in [self.x, self.y, self.z]
    
    def __reversed__(self) -> Iterator[float]:
        yield self.z
        yield self.y
        yield self.x

    def __eq__(self, other) -> bool:
        return isinstance(other, Vec3) and self.x == other.x and self.y == other.y and self.z == other.z

    def np(self) -> np.ndarray:
        """Convert to a numpy array"""
        return np.array(self)

    def magnitude(self) -> float:
        """Returns the distance Vec3 is from the zero vector.\n
        To get the distance two vectors are from each other you can use (self - other).magnitude()"""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize_or_zero(self) -> 'Vec3':
        """If the magnitude of self is zero, return zero vector.\n
        Otherwise returns a vector pointing the same direction as self but with a magnitude of 1"""
        mag = self.magnitude()
        if mag == 0:
            return Vec3.zero()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)
    
    @staticmethod
    def zero() -> 'Vec3':
        """Vec3(0,0,0)"""
        return Vec3(0,0,0)

    def normalize(self) -> 'Vec3':
        """ Returns a vector that points the same direction as self but has a magnitude of 1.\n
        !!NOTICE!! Self must not have a magnitude of 0. You can use normalize_or_zero() in that case."""
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def __repr__(self) -> str:
        return f"Vec3<{self.x}, {self.y}, {self.z}>"
    
    def cross(self, other: 'Vec3') -> 'Vec3':
        """Returns a vector perpendicular to both self and other.\n
        The magnitude of this vector is self.magnitude() * other.magnitude() * math.sin(Vec3(*self.get_rotation_to(other).as_rotvec()).magnitude())"""
        return Vec3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )

    def dot(self, other: 'Vec3') -> float:
        """Same as self.magnitude() * other.magnitude() * math.cos(Vec3(*self.get_rotation_to(other).as_rotvec()).magnitude())"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def practically_the_same_as(self, other: 'Vec3') -> bool:
        """If the self and other are < 0.0001 apart"""
        return (self - other).magnitude() < 1e-4
    
    def lerp(self, other: 'Vec3', t: float) -> 'Vec3':
        """Linear interpolation between start and end based on t."""
        return self * (1 - t) + other * t
    
    def transform(self, rotation: Rotation) -> 'Vec3':
        """Rotate a self by a rotation. You can also self * rotation to do the same thing."""
        np_result = rotation.apply(self.np())
        return Vec3(*np_result)
    
    def to_arbitrary_perpendicular(self) -> 'Vec3':
        """Returns a Vec3 with the same magnitude that is perpendicular to self.\n
        !!NOTICE!! It is not a continuous input does not yield a continuous output! You may need to use parallel transport to solve that."""
        if abs(self.x) < 0.00000000001 and abs(self.z) < 0.000000000001:
            arbitrary = Vec3.X()
        else:
            arbitrary = Vec3.Y()
        return self.cross(arbitrary).normalize() * self.magnitude()

    def to_arbitrary_rotation(self) -> Rotation:
        """Returns a rotation that faces the the direction of self.\n
        !!NOTICE!! A continuous input does not a continuous output! The 'roll' or 'up-direction' of the output is arbitrary.\n
        To prevent this use 'to_rotation_using_parallel_transport"""
        return Vec3.X().get_rotation_to(self)
    
    def get_rotation_to(self, other: 'Vec3') -> Rotation:
        """returns a rotation from self to a new vec3"""
        normalized_previous = self.normalize()
        normalized_new = other.normalize()

        if normalized_new.practically_the_same_as(normalized_previous):
            return Rotation.identity()
        
        if normalized_new.practically_the_same_as(-normalized_previous):
            axis = normalized_new.to_arbitrary_perpendicular()
            return Rotation.from_rotvec(axis.np() * math.pi) # something that is a complete flip
        
        rotation_axis = normalized_previous.cross(normalized_new).normalize() # sus
        rotation_angle = math.acos(normalized_previous.dot(normalized_new))

        return Rotation.from_rotvec(rotation_angle * np.array(rotation_axis))

    def to_rotation_using_parallel_transport(self, prev_rotation: Rotation, previous_tangent: 'Vec3') -> Rotation:
        """ returns a rotation that points in self direction, but only rolls the minimum amount\n
        the rotational transformation from previous_tangent to self (new_tangent) is applied to prev_rotation and that is returned"""
        return previous_tangent.get_rotation_to(self) * prev_rotation
    
    @staticmethod
    def random(seed, magnitude:float = 1) -> 'Vec3':
        """A returns a random rotation with a magnitude of the passed value\n
        seed can be _IntegerType | Generator | RandomState | None"""
        r = Rotation.random(1,seed)[0]
        return Vec3(magnitude,0,0) * r
    
    def to_euler_rotation(self, seq: str = "xyz") -> Rotation:
        """Uses the axises to be the values of an euler rotation"""
        return Rotation.from_euler(seq, self.np())

    def toDict(self) -> dict:
        """Returns {"x": ??, "y": ??, "z": ??} where the ?? are the values of self"""
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    @staticmethod
    def fromRotation(rotation: Rotation) -> 'Vec3':
        """ Get a normalized vector that points the direction of rotation\n
        Literally just Vec3.X() * rotation"""
        return Vec3.X() * rotation
    
    @staticmethod
    def fromPlaneFitting(inputPositions: List['Vec3']) -> 'Vec3':
        """From a set of points fits them to a plane a returns a normalized vector which is the normal of the plane\n
        idk how it works but it does. I just copied the algorithm from stack overflow.\n
        Might not be the most performant way of doing it but it works and I can't improve math that is over my head.\n
        source: https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
        """
        npInputPositions: List[np.ndarray] = []
        for pos in inputPositions:
            npInputPositions.append(pos.np())
        points = np.array(npInputPositions).T

        svd = np.linalg.svd(points - np.mean(points, axis = 1, keepdims=True))
        left = svd[0]
        result = left[:, -1]

        return Vec3(*result)

    def get_tangent_to(self, ahead: 'Vec3') -> 'Vec3':
        """If self and ahead are points in 3d space, returns a normalized or zero vector pointing the direction from self to ahead."""
        return (ahead - self).normalize_or_zero()
    
    def get_tangent_to_and_from(self, ahead: 'Vec3', behind: 'Vec3') -> 'Vec3':
        """If self, ahead, and behind are points in 3d space, returns a normalized or zero vector point the direction averaged between the direction from behind to self and self to ahead."""
        return (self.get_tangent_to(ahead) + behind.get_tangent_to(self)).normalize_or_zero()
    
    def get_tangent_optional_inputs(self, ahead: Optional['Vec3'], behind: Optional['Vec3']) -> 'Vec3':
        """Same as get_tangent_to_and_from but the inputs are optional. Must be supplied at least one of ahead and behind however."""
        if ahead is None:
            if behind is None:
                raise TypeError("Either ahead or behind or both need to exist to find tangent")
            else:
                return behind.get_tangent_to(self)
        else:
            if behind is None:
                return self.get_tangent_to(ahead)
            else:
                return self.get_tangent_to_and_from(ahead, behind)
            
    def get_normal_from_around(self, ahead: Optional['Vec3'], behind: Optional['Vec3'], left: Optional['Vec3'], right: Optional['Vec3']) -> 'Vec3':
        """Returns a predicted normal based on 4 optional locations around self.\n
        Note: at least one of ahead and behind must exist and at least one of left and right must exist."""
        if ahead is None and behind is None:
            raise TypeError("Either ahead or behind or both need to exist to find normal")
        if left is None and right is None:
            raise TypeError("Either left or right or both need to exist to find normal")
        tangent = self.get_tangent_optional_inputs(ahead,behind)
        binormal = self.get_tangent_optional_inputs(left, right)
        normal = binormal.cross(tangent).normalize_or_zero()
        return normal
    
    def rotate_around(self, axis: 'Vec3', rotation: float) -> 'Vec3':
        """Returns a Vec3 of self rotated around axis by rotation (in radians)"""
        return self * Rotation.from_rotvec((axis.normalize_or_zero() * rotation).np())
    
    @staticmethod
    def X(value: float = 1) -> 'Vec3':
        """short for Vec3(1,0,0) or Vec3(value,0,0)"""
        return Vec3(value,0,0)
    
    @staticmethod
    def Y(value: float = 1) -> 'Vec3':
        """short for Vec3(0,1,0) or Vec3(0,value,0)"""
        return Vec3(0,value,0)
    
    @staticmethod
    def Z(value: float = 1) -> 'Vec3':
        """short for Vec3(0,0,1) or Vec3(0,0,value)"""
        return Vec3(0,0,value)