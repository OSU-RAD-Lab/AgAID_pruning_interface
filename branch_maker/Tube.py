"""
Tube is a bunch of rings in a row - just List[Ring] plus some helper methods
"""

import copy
from typing import Iterable, Iterator, List, Optional, Tuple
import math
from Vec3 import Vec3
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from Interpolate import *
from Ring import Ring



class Tube:
    """A bunch of rings in a row - just List[Ring] plus some helper methods"""
    rings: List[Ring]
    """a list of rings. note: you can do `tube[i]` instead of `tube.rings[i]`"""
    lengths: List[float]


    def __init__(self, rings: List[Ring]):
        self.rings = rings
        self.compute_lengths()


    def total_length(self) -> float:
        if self.is_empty(): return 0
        return self.lengths[-1]


    def searchForDistance(self, dis: float) -> int:
        """binary search through self.lengths to largest length smaller than it. Returns the index of that. If its too big then it just returns the last index. negative dis will just give you the first index"""
        left, right = 0, len(self.lengths) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2

            if self.lengths[mid] < dis:
                result = mid
                left = mid + 1
            else:
                right = mid - 1

        return result


    def sampleAlong(self, dis: float) -> 'Ring':
        """returns a new ring based on rings around the place that the ring would be between the other rings around it as a weighted average. So like a virtual ring - if there was a ring at this dis then this is probably what it would look like"""
        lower_index = self.searchForDistance(dis)
        upper_index: int
        transition: float
        if lower_index == len(self) - 1:
            upper_index = lower_index
            transition = 0
        else:
            upper_index = lower_index + 1
            dis_to_lower = dis - self.lengths[lower_index]
            total_dis = self.lengths[lower_index + 1] - self.lengths[lower_index]
            transition = dis_to_lower / total_dis
        below_ring = self[lower_index]
        above_ring = self[upper_index]
        return below_ring.lerp(above_ring, transition)


    def is_empty(self) -> bool:
        return len(self) == 0


    def compute_lengths(self) -> None:
        """Every item is the distance from a ring (of relative index in self.branch_tube) to the beginning. The length follows the path. Useful for keeping things actually spaced apart correctly - because any two rings can be any distance apart"""
        self.lengths = []
        if (self.is_empty()): return
        current = 0
        previous: Vec3 = self[0].center
        for ring in self:
            current += (ring.center - previous).magnitude()
            self.lengths.append(current)
            previous = ring.center


    def __iter__(self) -> Iterator[Ring]:
        """`for ring in tube` looks nicer than `for ring in tube.rings`"""
        for i in range(len(self.rings)):
            yield self.rings[i]


    def zip(self) -> Iterable[Tuple[Ring, float]]:
        """hehe get zipped \n
        useful for `for ring, float in tube.zip()`"""
        return zip(self.rings, self.lengths)


    def __getitem__(self, index: int) -> Ring:
        """`tube[i]` looks nicer than `tube.rings[i]`"""
        return self.rings[index]
    

    def __len__(self) -> int:
        """`len(tube)` looks nicer than `len(tube.rings)`"""
        return len(self.rings)
    

    def append(self, other: Ring) -> None:
        """`tube.append(ring)` looks nicer than `tube.rings.append(ring)`"""
        if self.is_empty():
            self.lengths.append(0)
        else:
            self.lengths.append((other.center - self[-1].center).magnitude())
        self.rings.append(other)
        

    @staticmethod
    def from_verts(verts: List[Vec3], radius: float, connected: bool = False) -> 'Tube':
        """make a tube out of positions with a consistent radius and applies parallel transport for you - so all the rings have correct rotations as well. Connected is true if the front of the tube connects to back (like a torus)"""
        tube = Tube([Ring(vert, radius) for vert in verts])
        tube.apply_parallel_transport(connected)
        return tube


    def apply_parallel_transport(self, connected: bool = False, useExisting = False) -> None:
        """Modifies this tube directly
        Sets all the rings in the list so they face each other correctly
        This method only ensure that A) all rings face each. B) relative rotation around tangential axis between rings is minimized.
        Notice: initial rotation along the tangential axis is still chosen arbitrarily
        'connected' means that the first and last half look at each other - in addition to each's neighbor
        if its off then those two rings will point 100% directly at their only neighbors
        if a ring already has a rotation set, it will be use that existing rotation relative to the parallel transport"""
        previous_tangent: Optional[Vec3] = None
        previous_rotation:Optional[Rotation] = None

        for ring_index, ring in enumerate(self.rings):
            # save the existing rotation if it already has it
            extra_rotation: Optional[Rotation] = ring.rotation_or_none

            front:Optional[Vec3] = None
            behind:Optional[Vec3] = None

            if connected:
                behind = self.rings[(ring_index-1)%len(self.rings)].center
                front = self.rings[(ring_index+1)%len(self.rings)].center
            else:
                if ring_index != 0:
                    behind = self.rings[ring_index-1].center
                if ring_index != len(self.rings) - 1:
                    front = self.rings[ring_index+1].center

            # a vector that points the direction the ring should face based on the rings in front and behind it
            tangent_vector = ring.center.get_tangent_optional_inputs(front, behind) #ring.get_tangent_from_adjacent(front, behind)

            # this is probably only false the first iteration of the loop
            if previous_rotation is not None and previous_tangent is not None:
                # parallel transport works such that it takes the change in tangent vector from the previous ring to the current ring and converts that to a rotation - then it applies that rotation to the previous rotation
                # ie it uses change in tangent to determine change in rotation and uses that on previous rotation to make new rotation
                ring.rotation = tangent_vector.to_rotation_using_parallel_transport(previous_rotation, previous_tangent)
            else:
                # parallel transport requires the tangent and rotation of the previous ring - we dont have those (the previous tangent would be compute able, but the to get the rotation you would be in the same situation - needing the prev rotation)
                # so in order to resolve this issue, this method generates an arbitrary rotation facing the tangent vector. Whats arbitrary? the rotation around the tangent vector. (What's up? literally)
                ring.rotation = tangent_vector.to_arbitrary_rotation()

            previous_rotation = ring.rotation
            previous_tangent = tangent_vector

            if useExisting and extra_rotation is not None:
                ring.rotation = ring.rotation * extra_rotation


    def subdivide(self) -> 'Tube':
        """Returns a new tube
        A modified version of catmull clark for tubes, returns an entirely new tube that is a subdivided version of this one
        See `subdivision surface of cylinders.txt` for derivation of the equations used for this based on the original catmull clark algorithm
        Generally, the first and last tube remain unchanged (ie connected: bool is not implemented here yet and would effectively be false rn) - but all the rings get new rings inserted between then and the original middle tubes get altered to aline with those better
        This is supposed to produce a result as if the subdivision surface algorithm was applied once to a mesh created from this tube with infinite verts per ring
        """
        # 1. add first copied
        # 2. current is first (in the math its named Dc)
        # 3. evaluate and add between (in the math its named Dcn')
        # 4. increment current
        # 5. if current is last go to step 8
        # 6. evaluate and add replacement (in the math its named Dc')
        # 7. go to step 3
        # 8. add last copied
        # 9. return all that were added

        result = Tube([])

        # 1. add first copied
        first = self.rings[0]
        result.append(copy.deepcopy(first))

        # 2. current is first (in the math its named Dc)
        index = 0
        previous:Ring
        current:Ring = self.rings[index]
        next:Ring = self.rings[index + 1]
        previous_relative_top:Vec3
        current_relative_top:Vec3 = Vec3(0,current.radius,0).transform(current.rotation) 
        next_relative_top:Vec3 = Vec3(0,next.radius,0).transform(next.rotation)
        previous_relative_side:Vec3
        current_relative_side:Vec3 = Vec3(0,0,current.radius).transform(current.rotation)
        next_relative_side:Vec3 = Vec3(0,0,next.radius).transform(next.rotation)
        while True:
            # 3. evaluate and add between (in the math its named Dcn')
            between_center = (current.center + next.center) / 2
            between_radius = (current_relative_top + next_relative_top).magnitude() / 2
            between_normal = (current_relative_top + next_relative_top).normalize()
            between_binormal = (current_relative_side + next_relative_side).normalize()
            between_tangent = between_normal.cross(between_binormal).normalize()
            between_matrix = np.array([between_tangent.np(),between_normal.np(),between_binormal.np()]).T
            between_direction = Rotation.from_matrix(between_matrix)
            between = Ring(between_center, between_radius, between_direction)
            result.append(between)

            # 4. increment current
            index += 1
            # 5. if current is last go to step 8
            if index == len(self.rings) - 1: break
            previous = current
            previous_relative_top = current_relative_top
            previous_relative_side = current_relative_side
            current = next
            current_relative_top = next_relative_top
            current_relative_side = next_relative_side
            next = next = self.rings[index + 1]
            next_relative_top = Vec3(0,next.radius,0).transform(next.rotation)
            next_relative_side = Vec3(0,0,next.radius).transform(next.rotation)

            # 6. evaluate and add replacement (in the math its named Dc')
            replacement_center = (previous.center + current.center * 6 + next.center) / 8
            replacement_radius = (previous_relative_top + current_relative_top * 6 + next_relative_top).magnitude() / 8
            replacement_normal = (previous_relative_top + current_relative_top * 6 + next_relative_top).normalize()
            replacement_binormal = (previous_relative_side + current_relative_side * 6 + next_relative_side).normalize()
            replacement_tangent = replacement_normal.cross(replacement_binormal).normalize()
            replacement_matrix = np.array([replacement_tangent.np(),replacement_normal.np(),replacement_binormal.np()]).T
            replacement_direction = Rotation.from_matrix(replacement_matrix)
            replacement = Ring(replacement_center, replacement_radius, replacement_direction)
            result.append(replacement)
            # 7. go to step 3

        # 8. add last copied
        last = self.rings[len(self.rings) - 1]
        result.append(copy.deepcopy(last))

        result.compute_lengths()
        # 9. return all that were added
        return result
    

    def translate(self, position: Vec3) -> None:
        for ring in self:
            ring.center += position


    def scale(self, amount: float) -> None:
        for index, ring in enumerate(self):
            ring.center *= amount
            self.lengths[index] *= amount


    def rotate(self, rotation: Rotation) -> None:
        for ring in self:
            ring.center *= rotation
            ring.rotation *= rotation


    def simplify_angles(self, threshold: float = 0.0174) -> int:
        """removes rings that do not modify the dictions of the over thing much
        only considers the center of a ring, not its radius or direction
        threshold is in radians. A good range of values is pi/180 to pi/6
        returns the number removed
        the default threshold is pi/180, improvements are imperceptible beyond that"""

        cos_threshold = math.cos(threshold)

        removalCount = 0

        index = 1
        while (index < len(self) - 1): # skips first and last, simplify_angles will never remove those
            original_behind_orientation = (self[index].center - self[index-1].center).normalize_or_zero()
            new_behind_orientation = (self[index+1].center - self[index-1].center).normalize_or_zero()
            original_front_orientation = (self[index].center - self[index+1].center).normalize_or_zero()
            new_front_orientation = -new_behind_orientation
            alignment = min(original_behind_orientation.dot(new_behind_orientation), original_front_orientation.dot(new_front_orientation))

            if alignment > cos_threshold:
                self.rings.pop(index)
                removalCount +=1
                index-=2
            index+=1 # it will purposely skip an extra if one is removed

        self.compute_lengths()
        return removalCount
    
    def simplify_distances(self, threshold: float = 0.5) -> int:
        """removes rings that are close to each other
        threshold is a fraction of the radii
        only considers the center of a ring, not its radius or direction
        returns the number removed"""

        removalCount = 0

        index = 2
        first_index = 1
        while (index < len(self) - 1): # skips first and last, simplify_distances will never remove those
            # check if its in range
            distance = (self[index].center - self[first_index].center).magnitude()
            trueThreshold = threshold * (self[index].radius + self[first_index].radius) / 2
            if distance < trueThreshold: # gonna remove
                index += 1
            else: # done checking things to remove gonna merge first and last (index-1) then set first_index to index
                replacement = self[first_index].lerp(self[index-1],0.5)
                self.rings[first_index] = replacement
                for _ in range(index - (first_index + 1)):
                    self.rings.pop(first_index + 1)
                    removalCount+=1
                    index -= 1
                index += 1
                first_index += 1
        self.compute_lengths()
        return removalCount

    def __repr__(self) -> str:
        result = "Tube<"
        for index, ring in enumerate(self.rings):
            result += f"\n\t{ring} length:{self.lengths[index]},"
        result += "\n>"
        return result
    

    def toDictList(self) -> List[dict]:
        result: List[dict] = []
        for (ring, length) in self.zip():
            currentRingDict = ring.toDict()
            currentRingDict["length"] = length
            result.append(currentRingDict)
        
        return result