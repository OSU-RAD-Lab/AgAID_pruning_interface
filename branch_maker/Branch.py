import json
from typing import List, Tuple
import math
import numpy as np
from Vec3 import Vec3
from Ring import Ring
from Tube import Tube
from ObjWriter import ObjWriter, EndConnectionMode
from Interpolate import *

PHI = 1.61803399 # golden ratio
PHI_PHI = PHI * PHI # has some of the same properties as the golden ratio, but is a different number


def writeToJson(output: dict, path: str):
    """puts diction into a json file on the computer hard drive - i know its really quite magical"""
    with open(path, mode="w", encoding="utf-8") as fp:
        json.dump(output, fp)


class Branch():
    metadata: dict
    """what is written to the json"""
    children: List['Branch']
    """other branches that are splits off this one (if len 0 its a bud)"""
    tube: Tube
    """The tube has flavor"""
    bud_counts: List[Tuple[float,int]]
    """
    how many buds are at different locations.
    ordered by first element which is position along.
    The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)
    
    example: [(0.1,3),(0.2,2)]
    means: cutting anywhere from 0.0 to 0.1 removes 3 buds, cutting from 0.1 to 0.2 removes 2 buds, and cutting anywhere beyond that removes 1
    """


    def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
        position = start_position
        direction = direction.normalize()
        self.children = []
        ringIdOfChildren: List[int] = []
        energy = start_energy
        scale = 0.06 # multiplied by radii and positions as rings are created

        rand = np.random.default_rng(seed)

        # You: "Why is this code the way it is? Why are the numbers what they are? Where are the equations from?"
        # Answer: "I tried twice to be systematic about branch generation. The results were always limited.
        #   I came to the realization this is more of an artistic process and must be treated as such.
        #   Attempting to have a solid reason for every decision got in the way of the creation of the branches
        #   looking accurate and aesthetically pleasing. My design process here was more freeform than most programming.
        #   Similar to a cook adding spices to a dish, I generated some branches then noticed something I thought
        #   could be improved about the branches then and added a +0.2 here or math.sqrt() there. Some decisions
        #   have a solid reason behind them. Some are just because they 'looked better'. Most are somewhere in between."
        # You: "LMAO, that's some certified copium if I've ever seen any. This code is shit and calling it 'artistic'
        #   doesn't change that fact. You added extensive comments, but the underlying code is still terrible."
        # Answer: "Aaaaa why r u so mean :( The justification I gave is at least semi valid. And to make
        #   it better at this point it would require a fourth complete rewrite. I don't have time for
        #   that - I don't even have time to finish parameterizing this one."

        def calc_radius() -> float:
            # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
            # +0.5 to prevent division by 0
            vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
            energy_to_radius_converter = 0.055
            min_radius = 0.2 # because a branch a radius of 0 looks bad
            return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
        radius = calc_radius()

        def calc_length() -> float:
            # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
            length_candidate = lerp(0.5,1,rand.random()) * vigor
            min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
            return min(length_candidate, min_length) 

        self.tube = Tube([Ring(start_position*scale, radius*scale)])

        i = 0
        while True:
            radius = calc_radius()
            length = calc_length()
            position = (position + direction * length) # move along direction by length
            self.tube.append(Ring(position*scale, radius*scale))

            energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
            if energy < 0: # when its outta energy it finishes
                # End
                self.tube.compute_lengths()
                self.bud_counts = []
                bud_count = 1 # every branch has a bud at the tip
                for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
                    ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
                    ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
                    length = self.tube.lengths[ringIAfterSubDivide]
                    bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
                    self.bud_counts.append((length,bud_count)) 
                self.bud_counts.reverse() # tip->base => base->tip
                self.bud_counts.append((self.tube.total_length(), 1))
                # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
                # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
                self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
                self.tube.apply_parallel_transport() # make rings look at each other
                self.tube = self.tube.subdivide() # make tube look smoother
                self.tube.simplify_angles(0.05) # remove unnecessary geometry

                self.metadata = self.generateMetadata()
                return
            if energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
                # Preform Split
                splitAngle = rand.random() * math.tau
                randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

                newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
                newDirectionParent = (direction - randomNormal).normalize()

                energy -= radius # consumed expanding out
                distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
                energy -= distributed_energy

                new_child = Branch.__new__(Branch)
                self.children.append(new_child)
                new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
                    start_position=position,
                    direction=newDirectionChild,
                    vigor=exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
                    abundance=lerp(abundance,1,0.2), # children have more abundance
                    start_energy=distributed_energy
                )

                ringIdOfChildren.append(i)

                direction = newDirectionParent
                position += newDirectionParent * radius / 2 # divided by two so it does not stick out
            else:
                # Dont split just go straight and turn upwards a lil
                goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
                goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
                direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
                abundance = lerp(abundance,0,0.05)# less abundance
                vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
            i += 1


    def getTotalNumberOfBuds(self) -> int:
        if len(self.bud_counts) == 0:
            return 1 # every child is a bud rn
        else:
            return self.bud_counts[0][1] # first child, get count (not position)


    def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
        self.tube.compute_lengths()
        obj.writeComment(f"Tube @ {at}")
        obj.writeTube(self.tube, around_resolution, at=at, endConnectionMode=EndConnectionMode.CAP_BACK)
        for child in self.children:
            obj.writeComment(f"Starting Child")
            child.toObj(obj,at)


    @staticmethod
    def fromValue(value: float, direction: Vec3, seed: int, energy: float = 15) -> 'Branch':
        """
        value is from 0 to 1 and indicates. zero is typically too many buds, one is typically too vigorous.
        However value is often more of a suggestion - far from completely controlling it.
        directions more upward tend to need lower values to compensate and some seeds will always to vigorous regardless of value

        direction is the direction that the branch starts growing from.

        a change in seed changes the entire branch massively - but changing the other prams minimally changes it

        energy can be left at default 
        """
        result = Branch(
            abundance=  exerp(0.75,0.15,value),
            seed=       seed,
            direction=  direction,
            vigor=      exerp(0.2,3.5,value),
            start_energy=     energy
        )
        result.metadata["generation_settings"] = {
            "value": value,
            "direction": direction.normalize().toDict(),
            "seed": seed,
            "energy": energy
        }
        return result
    

    def generateMetadata(self) -> dict:
        result: dict = {
            "tube": self.tube.toDictList(),
            "bud_counts": self.bud_counts,
            "children": []
        }
        for index, child in enumerate(self.children):
            childData = child.generateMetadata()
            childData["location"] = self.bud_counts[index][0]
            result["children"].append(childData)
        return result


    # def generateTubeDictData(self) -> List[List[dict]]:
    #     result: List[List[dict]] = [self.tube.toDictList()]
    #     if len(self.children) != 0:
    #         for child in self.children:
    #             result += child.generateTubeDictData()
    #     return result


    def writeEverything(self, name = "output") -> None:
        "Exports the branch as a obj and as a json file all in one go. \n\n Isn't that so cool?"
        with open(f"{name}.obj", "w") as fp:
            obj = ObjWriter(fp, name, True)
            self.toObj(obj)
        writeToJson(self.metadata, f"{name}.obj.json")