import copy
import random
import json
from typing import Any, Callable, Generator, List, Optional, Tuple
import math

import numpy as np
from Vec3 import Vec3
from Ring import Ring
from Tube import Tube 
from ObjWriter import ObjWriter, EndConnectionMode
from scipy.spatial.transform import Rotation
R = Rotation # for those who prefer this. I like Rotation better tho

PHI = 1.61803399 # golden ratio
PHI_PHI = PHI * PHI # has some of the same properties as the golden ratio, but is a different number

def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation between start and end as t goes from 0 to 1."""
    return start * (1 - t) + end * t

def inverse_lerp(start: float, end: float, value: float) -> float:
    """linear interpolates 0 to 1 as value goes from start to end"""
    return (value-start)/(end-start)

def exerp(start: float, end: float, t: float) -> float:
    """exponential interpolation between start and end based on t."""
    return start*pow(end/start,t)


def sample(start: float, end: float, resolutionAlong: int, equation:  Callable[[float], Any]) -> List[Any]:
    """make a list of values by calling a callable resolutionAlong time starting from start and going to end all those collected into a List is returned"""
    points = []
    for i in range(resolutionAlong):
        value = lerp(start, end, i/float(resolutionAlong))
        points.append(equation(value))
    return points


def randomized_branch_curve_parametric(seed: int, prams: List[Tuple[float,float]], t:float) -> Vec3:
    """for a given seed and prams, this function is a parametric equation based on t that provides somewhat natural looking randomized movements
    seed is the seed for the randomization
    prams is a list of size 2 tuples which determines the character of the random movements
    ever element of prams is a random movement - specific a vector of length [random number between sqrt(3) to 2sqrt(3)] times the first value of the tuple starts with a random rotations and rotates around each of the x,y,z axises at a rate of [random number between 1-2 times the second value of the tuple] radians per increment of t
    All the 'random' movements added by each element are added together and that is the result
    The first (0) pram is known as the frequency and the second (1) as the magnitude. It is recommend to use a combination of higher magnitude but lower frequency and lower magnitude but higher frequency for more accurate results.
    Recomputing the randoms every sample is horribly unoptimized - but its fine, the time it takes to run this for a branch is far outshadowed by writing the branch to the obj. If your not as lazy as me, integrating this function with the 'sample' function would not be that hard"""
    random.seed(seed)
    result = Vec3.zero()
    rot_base = Rotation.random(len(prams), seed)
    for index, i in enumerate(prams):
        result += Vec3(random.uniform(i[1], i[1]*2),random.uniform(i[1], i[1]*2),random.uniform(i[1], i[1]*2)) * Rotation.from_euler("xyz",[t*random.uniform(i[0], i[0]*2),t*random.uniform(i[0], i[0]*2),t*random.uniform(i[0], i[0]*2)])* rot_base[index]
    return result


def writeToJson(output: dict, path: str):
    """puts diction into a json file on the computer hard drive - i know its really quite magical"""
    with open(path, mode="w", encoding="utf-8") as fp:
        json.dump(output, fp)

class Branch:
    branch_tube: 'Tube'
    """The main body of the Branch"""
    bud_shape: 'Tube'
    """tube that is copied/referenced to make all the bud geometries"""
    minimum_along: float
    """the first value that the equation is sampled at
    always 0 cause i haven't had a reason to change it"""
    maximum_along: float
    """the last value that the equation is sampled at
    always 0 cause i haven't had a reason to change it"""
    along_resolution: int
    """how many rings are used in this tube"""
    branch_around_resolution: int
    """how many verts per ring are used for the branch body"""
    bud_around_resolution: int
    """how many verts per ring are used on the buds"""
    max_bud_spacing: float
    """in meters"""
    min_bud_spacing: float
    """in meters"""
    bud_placement_seed: int
    """buds will be relatively the same spots even if you change other parameters. But if you change this one, they will be in very different spots. You most likely will want to change this at the same time as curve_seed"""
    bud_locations: List[Tuple[float, float]] # along (minimum_along-maximum_along) and around [0-math.tau)
    """used to store where on the branch the branch the buds are. First is along (0-1) so positive this and you move against the rings. Around is second which takes range [0-math.tau) which is the radians around the branch. (what is up is arbitrary - but consistent if the branch shape stays the same) This is set by place_bud_locations method"""

    def __init__(self, branch_tube: Tube, bud_tube: Optional[Tube] = None,
                 min_bud_spacing: float = 0.04, max_bud_spacing: float = 0.08, branch_around_resolution: int = 16, bud_around_resolution: int = 8,
                 bud_placement_seed: int = 0):
        """min_bud_spacing & max_bud_spacing
          in meters
        along_resolution
          how many rings are initially used in this tube (this number is reduced because the branch gets simplified)
        branch_around_resolution
          how many verts per ring are used for the branch body
        bud_around_resolution
          how many verts per ring are used on the buds
        bud_placement_seed
          buds will be relatively the same spots even if you change other parameters. But if you change this one, they will be in very different spots. You most likely will want to change this at the same time as curve_seed
        """


        if bud_tube is None:
            self.bud_shape = Tube([
                Ring(Vec3(0, -0.002, 0), 0.005),
                Ring(Vec3(0, 0.003, 0), 0.004),
                Ring(Vec3(0.008, 0.006, 0), 0.005),
                Ring(Vec3(0.015, 0.008, 0), 0.007),
                Ring(Vec3(0.02, 0.006, 0), 0.002)
            ])
            self.bud_shape.apply_parallel_transport()
            self.bud_shape = self.bud_shape.subdivide().subdivide()
            self.bud_shape.simplify_angles(0.06)
        else:
            self.bud_shape = bud_tube

        
        self.minimum_along = 0
        self.maximum_along = 1
        self.branch_around_resolution = branch_around_resolution
        self.bud_around_resolution = bud_around_resolution
        self.max_bud_spacing = max_bud_spacing
        self.min_bud_spacing = min_bud_spacing
        self.bud_placement_seed = bud_placement_seed
        self.branch_tube = branch_tube
        self.length = self.branch_tube.total_length()

        self.bud_locations = self.place_bud_locations()

    def getBudTube(self, along: float, around: float) -> 'Tube':
        """From a given along (0-1) and around [0-math.tau) find that spot on the branch and uses self.bud_shape to create a new tube that will be the bud with its location and radius and rotation all figured out real good"""
        sampled_ring = self.branch_tube.sampleAlong(along)
        sampled_ring.rotation = sampled_ring.rotation * Rotation.from_euler("xyz", [-around,0,0])
        surface = Vec3(0,sampled_ring.radius,0) * sampled_ring.rotation + sampled_ring.center
        modified_rings = []
        for ring in self.bud_shape:
            copied_ring = copy.copy(ring)
            copied_ring.center *= sampled_ring.rotation
            copied_ring.center += surface
            copied_ring.rotation = sampled_ring.rotation * copied_ring.rotation
            modified_rings.append(copied_ring)
        result = Tube(modified_rings)
        # parallel transport should have been applied on the self.bud_shape, so doing it here would make it no longer parallel and what ever is transport is kinda meaning less then - so nah
        return result
    
    def place_bud_locations(self) -> List[Tuple[float, float]]:
        """generates values usable by place_bud_objects based on max_bud_spacing, min_bud_spacing, and bud_placement_seed, this goes in bud_locations. The first item of the tuple is along (0-1) down the rings, and the second item [0-math.tau) is around a given ring."""
        returnList = []

        VAR_DIS_RANGE = self.max_bud_spacing - self.min_bud_spacing

        # two values that are seemingly random between 0 and 1
        disRandState = (self.bud_placement_seed * PHI) % 1
        currentRot = (self.bud_placement_seed * PHI_PHI) % 1

        currentDis = self.min_bud_spacing

        while currentDis < self.length:
            returnList.append((currentDis, math.tau * currentRot))

            currentRot = (PHI + currentRot) % 1 # correctly simulates how plants choose the direction to multiple things in in most cases
            disRandState = (PHI_PHI + disRandState) % 1 # similar to the one for rotation but not an attempt at realism just a way to get uniformly random numbers

            currentDis += self.min_bud_spacing + VAR_DIS_RANGE * disRandState

        return returnList

    def writeToObj(self, obj: ObjWriter, location: Vec3) -> None:
        """writes the body then all the buds. It uses CAP_BACK EndConnectionMode. What else can I tell you that you can clearly see by glancing at the code"""
        obj.writeComment(f"Branch @ {location}\n")
        obj.writeTube(self.branch_tube, self.branch_around_resolution, EndConnectionMode.CAP_BACK, location)
        for index, bud in enumerate(self.bud_locations):
            obj.writeComment(f"Bud {index} Branch @ {location}")
            obj.writeTube(self.getBudTube(bud[0], bud[1]), self.bud_around_resolution,EndConnectionMode.CAP_BACK, location)
    
    def writeToDict(self) -> dict:
        """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it. Using this method for any purpose other that exporting to a json would be pretty cringe"""
        rings = []
        for ring in self.branch_tube:
            rings.append(ring.toDict())

        buds = []
        for bud in self.bud_locations:
            buds.append({
                "along":bud[0],
                "around":bud[1]
            })

        return {
            "rings": rings,
            "buds": buds,
            "settings": {
                "branch_around_resolution": self.branch_around_resolution,
                "bud_around_resolution":self.bud_around_resolution,
                "max_bud_spacing": self.max_bud_spacing,
                "min_bud_spacing": self.min_bud_spacing,
                "bud_placement_seed": self.bud_placement_seed,
                "length": self.length,
                "type":"Branch"
            }
        }
    
    def write_all(self, path: str):
        """does all the exporting shenanigans for you all bundled up in one lil method call. """
        with open(f"{path}.obj", "w") as fp:
            obj = ObjWriter(fp, title=path, triangulate=True)
            self.writeToObj(obj, Vec3(0,0,0))
        writeToJson(self.writeToDict(), f"{path}.obj.json")


class ParametricBranch(Branch):
    equation: Callable[[float], 'Ring']
    """Parametric Equation for getting the initial locations of the rings"""
    length: float
    """ in meters
    length from one ring to another in branch tube. The actual value is from the start - but it follows the rings - so not straight"""
    curve_prams: List[Tuple[float,float]]
    """The curve is determined by a couple other things, but this gives the detail. Each element adds some detail to the branch
    Each element adds a random swing around rotation. There are two keep facts about that - the frequency or speed which it swings around which determines the how much detail along it there is and the magnitude (how much effect it has on the result) It is recommended to use low magnitudes with higher frequencies to keep it from looking stupid
    The first item of the tuple is frequency in 1-2 radians/per length of branch (1-2 cause random) the second is in 1-2 meters. 
    """
    start_radius: float
    """in meters"""
    end_radius: float
    """in meters"""
    curve_seed: int
    """used to seed the random for how the branch is shaped. Different seed - different branch shape - same seed same branch shape for the most part"""
    def __init__(self,
                 min_bud_spacing: float = 0.04, max_bud_spacing: float = 0.08,
                 along_resolution: int = 128, branch_around_resolution: int = 16, bud_around_resolution: int = 8,
                 bud_placement_seed: int = 0, curve_seed: int = 0,
                 length: float = 1, curve_prams: List[Tuple[float,float]] = [(0.6,.25), (1.8,0.05), (3.3,0.02)],
                 start_radius: float = 0.015, end_radius:float = 0.005):
        """min_bud_spacing & max_bud_spacing
          in meters
        along_resolution
          how many rings are initially used in this tube (this number is reduced because the branch gets simplified)
        branch_around_resolution
          how many verts per ring are used for the branch body
        bud_around_resolution
          how many verts per ring are used on the buds
        bud_placement_seed
          buds will be relatively the same spots even if you change other parameters. But if you change this one, they will be in very different spots. You most likely will want to change this at the same time as curve_seed
        curve_seed
          used to seed the random for how the branch is shaped. Different seed - different branch shape - same seed same branch shape for the most part
        length
          length from one ring to another in branch tube. The actual value is from the start - but it follows the rings - so not straight
        curve_prams
          The curve is determined by a couple other things, but this gives the detail. Each element adds some detail to the branch
          Each element adds a random swing around rotation. There are two keep facts about that - the frequency or speed which it swings around which determines the how much detail along it there is and the magnitude (how much effect it has on the result) It is recommended to use low magnitudes with higher frequencies to keep it from looking stupid
          The first item of the tuple is frequency in 1-2 radians/per length of branch (1-2 cause random) the second is in 1-2 meters.
        start_radius & end_radius
          in meters
        """


        self.length = length
        self.curve_seed = curve_seed
        self.curve_prams = curve_prams
        self.start_radius = start_radius
        self.end_radius = end_radius
        self.along_resolution = along_resolution

        self.equation = lambda t: Ring(
                            randomized_branch_curve_parametric(self.curve_seed,self.curve_prams,t) + Vec3(t,0,0),
                            lerp(self.start_radius*2,self.end_radius,t) * lerp(0.5,1,t) #  * lerp(0.5,1,t) counter balances the self.start_radius*2. This is included because it make the branch taper in a more aesthetically pleasing way in my personal opinion
                        )
        branch_tube = Tube(sample(0, 1, self.along_resolution, self.equation))
        self.along_resolution -= branch_tube.simplify_angles(0.02)
        branch_tube.translate(-branch_tube[0].center) # after sample
        scale_factor = length/branch_tube.total_length() # after compute_lengths
        branch_tube.scale(scale_factor)# after translate and uses scale_factor
        branch_tube.apply_parallel_transport() # after sample
        aligning_rotation = (branch_tube[self.along_resolution>>1].center.normalize_or_zero() + branch_tube[-1].center.normalize_or_zero()).get_rotation_to(Vec3.X()) # 
        branch_tube.rotate(aligning_rotation)  # after translate and apply_parallel_transport
        
        super().__init__(branch_tube,None, min_bud_spacing, max_bud_spacing, branch_around_resolution, bud_around_resolution, bud_placement_seed)

    @staticmethod
    def from_vigor_spacing(vigor: float = 0.5, spacing: float = 0.5, seed: Optional[int] = None) -> 'ParametricBranch':
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        length = exerp(0.25, 3, vigor)
        min_bud_spacing = length / exerp(1, 9, spacing)

        return ParametricBranch(min_bud_spacing=min_bud_spacing, max_bud_spacing=min_bud_spacing*2,
                      bud_placement_seed=seed, curve_seed=seed,
                      length=length, start_radius=exerp(0.006, 0.015, vigor), end_radius=0.004,
                      curve_prams=[(0.6,.25/length), (1.8,0.05/length), (3.3,0.02/length)])
    
    def writeToDict(self) -> dict:
        super_result = super().writeToDict()
        curve_prams = []
        for pram in self.curve_prams:
            curve_prams.append({
                "freq": pram[0],
                "mag": pram[1]
            })

        super_result["settings"]["curve_seed"] = self.curve_seed
        super_result["settings"]["curve_prams"] = self.curve_prams
        super_result["settings"]["start_radius"] = self.start_radius
        super_result["settings"]["end_radius"] = self.end_radius
        super_result["settings"]["along_resolution"] = self.along_resolution
        super_result["settings"]["type"] = "ParametricBranch"

        return super_result


class IterativeBranch(Branch):
    fStep: Vec3
    """How far each step of the simulation should go"""
    rStep: Rotation
    """How much default curvature is added. Kinda simulates gravity but not really. Makes it generally more curvy and more interesting"""
    maxMutation: float
    """The higher the number the larger the changes in direction will be"""
    mutationPow: float
    """The higher the number the the less often large changes in direction will occur"""
    goal: Vec3
    """Which direction the branch reaches towards"""
    iterations: int
    """The number of simulation steps"""
    subdivisionSteps: int
    """How many time to subdivide before smoothing. Int >=0 dont go too high (>=4) or it will be very slow for no benefit"""
    smoothing: float
    """in radians of minium allowed corner sharpness. Larger values are less smooth"""

    def __init__(self, seed: Optional[int] = None, starting_ring:Optional[Ring] = None,
                 length: float = 1, rStep: Rotation = Rotation.from_euler("xyz",[0,0.2,0]),
                 maxMutation: float = math.tau/3, mutationPow: float = 5,
                 goal: Vec3 = Vec3.Y(), iterations = 16,
                 subdivisionSteps: int = 3, smoothing: float = 0.04,
                 min_bud_spacing: float = 0.04, max_bud_spacing: float = 0.08,
                 branch_around_resolution: int = 16, bud_around_resolution: int = 8,
                 start_radius: float = 0.015, end_radius:float = 0.005):
        
        rand = np.random.default_rng(seed)
        
        self.seed = seed
        fStep = Vec3(length/iterations,0,0)
        self.fStep = fStep
        self.rStep = rStep
        self.maxMutation = maxMutation
        self.mutationPow = mutationPow
        self.goal = goal
        self.iterations = iterations
        self.subdivisionSteps = subdivisionSteps
        self.smoothing = smoothing
        self.start_radius = start_radius
        self.end_radius = end_radius



        # initial parameters

        if starting_ring is None:
            ring = Ring(Vec3(0,0,0),0.1,Rotation.random(1,rand)[0])
        else:
            ring = starting_ring
        self.starting_ring = ring
        tube = Tube([ring])
        iterationRemaining = iterations
        while iterationRemaining > 0:
            iterationRemaining-=1

            rMutation = math.pow(rand.random(),mutationPow) * maxMutation #* pow((1-Vec3.fromRotation(ring.rotation).dot(goal))/2,0.1)

            original_correlation = (2-Vec3.fromRotation(ring.rotation).dot(goal))/3

            # new possible directions
            optionA = ring.rotation * Vec3.random(rand, rMutation).to_euler_rotation() * rStep
            optionB = ring.rotation * Vec3.random(rand, rMutation).to_euler_rotation() * rStep
            optionC = ring.rotation * Vec3.random(rand, rMutation).to_euler_rotation() * rStep
            optionD = ring.rotation * Vec3.random(rand, rMutation).to_euler_rotation() * rStep
            # optionE = ring.rotation * Vec3.random(rMutation * pow(original_correlation,4), rand).to_euler_rotation() * (rStep)
            # optionF = ring.rotation * Vec3.random(rMutation * pow(original_correlation,0.33), rand).to_euler_rotation() * (rStep)
            # optionG = ring.rotation * Vec3.random(rMutation * pow(original_correlation,3), rand).to_euler_rotation() * (rStep.inv())
            # optionH = ring.rotation * Vec3.random(rMutation * pow(original_correlation,0.75), rand).to_euler_rotation() * (rStep)



            # pick the direction the the most alignment with the goal
            dotA = (Vec3.X() * optionA).dot(goal)
            dotB = (Vec3.X() * optionB).dot(goal)
            dotC = (Vec3.X() * optionC).dot(goal)
            dotD = (Vec3.X() * optionD).dot(goal)
            # dotE = (Vec3.X() * optionE).dot(goal)
            # dotF = (Vec3.X() * optionF).dot(goal)
            # dotG = (Vec3.X() * optionG).dot(goal)
            # dotH = (Vec3.X() * optionH).dot(goal)

            if dotA > dotB:
                dotAB = dotA
                optionAB = optionA
            else:
                dotAB = dotB
                optionAB = optionB
            if dotC > dotD:
                dotCD = dotC
                optionCD = optionC
            else:
                dotCD = dotD
                optionCD = optionD
            # if dotE > dotF:
            #     dotEF = dotE
            #     optionEF = optionE
            # else:
            #     dotEF = dotF
            #     optionEF = optionF
            # if dotG > dotH:
            #     dotGH = dotG
            #     optionGH = optionG
            # else:
            #     dotGH = dotH
            #     optionGH = optionH
            if dotAB > dotCD:
                dotABCD = dotAB
                optionABCD = optionAB
            else:
                dotABCD = dotCD
                optionABCD = optionCD
            # if dotEF > dotGH:
            #     dotEFGH = dotEF
            #     optionEFGH = optionEF
            # else:
            #     dotEFGH = dotGH
            #     optionEFGH = optionGH
            # if dotABCD > dotEFGH:
            choice = optionABCD
            # else:
            #     choice = optionEFGH

            # choice: Rotation
            # if dotA > dotB:
            #     if dotA > dotC:
            #         choice = optionA
            #         print("A")
            #     else:
            #         choice = optionC
            #         print("C")
            # else:
            #     if dotB > dotC:
            #         choice = optionB
            #         print("B")
            #     else:
            #         choice = optionC
            #         print("C")
            
            ring = Ring(ring.center + fStep * choice, lerp(0.05,0.1,iterationRemaining/iterations)*lerp(0.8,1,iterationRemaining/iterations), choice)
            tube.append(ring)

        for index, ring in enumerate(tube):
            t = index/iterations
            radius = lerp(start_radius*2,end_radius,t) * lerp(0.5,1,t)
            ring.radius = radius

        # make it beautiful
        tube.apply_parallel_transport()

        for _ in range(subdivisionSteps):
            tube = tube.subdivide()
        
        tube.simplify_angles(smoothing)

        super().__init__(tube, None, min_bud_spacing, max_bud_spacing, branch_around_resolution, bud_around_resolution, seed or random.randint(0,2**32-1))
        
    def writeToDict(self) -> dict:
        super_result = super().writeToDict()
        super_result["settings"]["seed"] = self.seed
        rStepAsQuat = self.rStep.as_quat()
        rStepAsDict =  {
            "w": rStepAsQuat[0],
            "x": rStepAsQuat[1],
            "y": rStepAsQuat[2],
            "z": rStepAsQuat[3]
        }
        super_result["settings"]["rStep"] = rStepAsDict
        super_result["settings"]["maxMutation"] = self.maxMutation
        super_result["settings"]["type"] = "IterativeBranch"
        super_result["settings"]["goal"] = self.goal.toDict()
        super_result["settings"]["iterations"] = self.iterations
        super_result["settings"]["subdivisionSteps"] = self.subdivisionSteps
        super_result["settings"]["smoothing"] = self.smoothing
        super_result["settings"]["starting_ring"] = self.starting_ring.toDict()
        super_result["settings"]["start_radius"] = self.start_radius
        super_result["settings"]["end_radius"] = self.end_radius

        return super_result
    
    # @staticmethod
    # def from_vigor_spacing(vigor: float = 0.5, spacing: float = 0.5, seed: Optional[int] = None) -> 'IterativeBranch':
    #     if seed is None:
    #         seed = random.randint(0, 2**32 - 1)
        
    #     length = exerp(0.25, 3, vigor)
    #     min_bud_spacing = length / exerp(1, 9, spacing)
    #     iterations = int(vigor*15)+10

    #     return IterativeBranch(seed=seed,iterations=iterations,length=length,min_bud_spacing=min_bud_spacing, max_bud_spacing=min_bud_spacing*2)


# # # # # # # # # # # # # # actual
class BetterBranch():
    children: List['BetterBranch'] # (if len 0 its a bud)
    tube: Tube
    bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

    def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
        position = start_position
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
                # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
                # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
                self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
                self.tube.apply_parallel_transport() # make rings look at each other
                self.tube = self.tube.subdivide() # make tube look smoother
                self.tube.simplify_angles(0.05) # remove unnecessary geometry


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


                new_child = BetterBranch.__new__(BetterBranch)
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
    def fromValue(value: float, direction: Vec3, seed: int, energy: float = 15) -> 'BetterBranch':
        return BetterBranch(
            abundance=  exerp(0.75,0.15,value),
            seed=       seed,
            direction=  direction,
            vigor=      exerp(0.2,3.5,value),
            start_energy=     energy
        )
    

    def generateTubeDictData(self) -> List[List[dict]]:
        result: List[List[dict]] = [self.tube.toDictList()]
        if len(self.children) != 0:
            for child in self.children:
                result += child.generateTubeDictData()
        return result
            

    def writeToDict(self) -> dict:
        """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
        return {
            "tube": [
                self.generateTubeDictData()
            ]
        }





















































BOBJ: ObjWriter
Boffset: int = 0
bigboy: 'BetterBranch'


# # 1
# class BetterBranch():
#     children: List['BetterBranch'] # (if len 0 its a bud)
#     tube: Tube
#     bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

#     def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
#         position = start_position
#         self.children = []
#         ringIdOfChildren: List[int] = []
#         energy = start_energy
#         scale = 0.06 # multiplied by radii and positions as rings are created

#         rand = np.random.default_rng(seed)

#         def calc_radius() -> float:
#             # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
#             # +0.5 to prevent division by 0
#             vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
#             energy_to_radius_converter = 0.055
#             min_radius = 0.2 # because a branch a radius of 0 looks bad
#             return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
#         radius = calc_radius()

#         def calc_length() -> float:
#             # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
#             length_candidate = lerp(0.5,1,rand.random()) * vigor
#             min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
#             return min(length_candidate, min_length) 

#         self.tube = Tube([Ring(start_position*scale, radius*scale, direction.to_arbitrary_rotation())])

#         global BOBJ
#         global Boffset
#         global bigboy
#         if Boffset == 0:
#             bigboy = self
#         bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#         Boffset += 1

#         i = 0
#         while i==0:
#             radius = calc_radius()
#             length = calc_length()
#             position = (position + direction * length) # move along direction by length
#             self.tube.append(Ring(position*scale, radius*scale, direction.to_arbitrary_rotation()))
            
#             bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             Boffset += 1
            
#             # energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
#             # if energy < 0: # when its outta energy it finishes
#             #     # End
#             #     self.tube.compute_lengths()
#             #     self.bud_counts = []
#             #     bud_count = 1 # every branch has a bud at the tip
#             #     for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
#             #         ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
#             #         ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
#             #         length = self.tube.lengths[ringIAfterSubDivide]
#             #         bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
#             #         self.bud_counts.append((length,bud_count)) 
#             #     self.bud_counts.reverse() # tip->base => base->tip
#             #     # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
#             #     # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
#             #     self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
#             #     self.tube.apply_parallel_transport() # make rings look at each other
#             #     self.tube = self.tube.subdivide() # make tube look smoother
#             #     self.tube.simplify_angles(0.05) # remove unnecessary geometry

#             #     bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             #     Boffset += 1

#             #     return
#             # if energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
#             #     # Preform Split
#             #     splitAngle = rand.random() * math.tau
#             #     randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

#             #     newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
#             #     newDirectionParent = (direction - randomNormal).normalize()

#             #     energy -= radius # consumed expanding out
#             #     distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
#             #     energy -= distributed_energy


#             #     new_child = BetterBranch.__new__(BetterBranch)
#             #     self.children.append(new_child)
#             #     new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
#             #         start_position=position,
#             #         direction=newDirectionChild,
#             #         vigor=exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
#             #         abundance=lerp(abundance,1,0.2), # children have more abundance
#             #         start_energy=distributed_energy
#             #     )

#             #     ringIdOfChildren.append(i)


#             #     direction = newDirectionParent
#             #     position += newDirectionParent * radius / 2 # divided by two so it does not stick out
#             # else:
#             #     # Dont split just go straight and turn upwards a lil
#             #     goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
#             #     goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
#             #     direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
#             #     abundance = lerp(abundance,0,0.05)# less abundance
#             #     vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
#             i += 1

#     def getTotalNumberOfBuds(self) -> int:
#         if len(self.bud_counts) == 0:
#             return 1 # every child is a bud rn
#         else:
#             return self.bud_counts[0][1] # first child, get count (not position)

#     def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
#         self.tube.compute_lengths()
#         obj.writeComment(f"Tube @ {at}")
#         obj.writeTube(self.tube, around_resolution, at=at, endConnectionMode=EndConnectionMode.CAPS)
#         for child in self.children:
#             obj.writeComment(f"Starting Child")
#             child.toObj(obj,at)


#     @staticmethod
#     def fromValue(value: float, direction: Vec3, seed: int) -> 'BetterBranch':
#         return BetterBranch(
#             abundance=  exerp(0.75,0.15,value),
#             seed=       seed,
#             direction=  direction,
#             vigor=      exerp(0.2,3.5,value),
#         )
    

#     def generateTubeDictData(self) -> List[List[dict]]:
#         result: List[List[dict]] = [self.tube.toDictList()]
#         if len(self.children) != 0:
#             for child in self.children:
#                 result += child.generateTubeDictData()
#         return result
            

#     def writeToDict(self) -> dict:
#         """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
#         return {
#             "tube": [
#                 self.generateTubeDictData()
#             ]
#         }
        



# # 2
# class BetterBranch():
#     children: List['BetterBranch'] # (if len 0 its a bud)
#     tube: Tube
#     bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

#     def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
#         position = start_position
#         self.children = []
#         ringIdOfChildren: List[int] = []
#         energy = start_energy
#         scale = 0.06 # multiplied by radii and positions as rings are created

#         rand = np.random.default_rng(seed)

#         def calc_radius() -> float:
#             # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
#             # +0.5 to prevent division by 0
#             vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
#             energy_to_radius_converter = 0.055
#             min_radius = 0.2 # because a branch a radius of 0 looks bad
#             return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
#         radius = calc_radius()

#         def calc_length() -> float:
#             # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
#             length_candidate = lerp(0.5,1,rand.random()) * vigor
#             min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
#             return min(length_candidate, min_length) 

#         self.tube = Tube([Ring(start_position*scale, radius*scale, direction.to_arbitrary_rotation())])

#         global BOBJ
#         global Boffset
#         global bigboy
#         if Boffset == 0:
#             bigboy = self
#         bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#         Boffset += 1

#         i = 0
#         while True:
#             radius = calc_radius()
#             length = calc_length()
#             position = (position + direction * length) # move along direction by length
#             self.tube.append(Ring(position*scale, radius*scale, direction.to_arbitrary_rotation()))
            
#             bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             Boffset += 1
            
#             energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
#             if energy < 0: # when its outta energy it finishes
#                 # End
#             #     self.tube.compute_lengths()
#             #     self.bud_counts = []
#             #     bud_count = 1 # every branch has a bud at the tip
#             #     for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
#             #         ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
#             #         ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
#             #         length = self.tube.lengths[ringIAfterSubDivide]
#             #         bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
#             #         self.bud_counts.append((length,bud_count)) 
#             #     self.bud_counts.reverse() # tip->base => base->tip
#             #     # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
#             #     # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
#             #     self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
#             #     self.tube.apply_parallel_transport() # make rings look at each other
#             #     self.tube = self.tube.subdivide() # make tube look smoother
#             #     self.tube.simplify_angles(0.05) # remove unnecessary geometry

#             #     bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             #     Boffset += 1

#                 return
#             # if False and energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
#             #     # Preform Split
#             #     pass
#             #     splitAngle = rand.random() * math.tau
#             #     randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

#             #     newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
#             #     newDirectionParent = (direction - randomNormal).normalize()

#             #     energy -= radius # consumed expanding out
#             #     distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
#             #     energy -= distributed_energy


#             #     new_child = BetterBranch.__new__(BetterBranch)
#             #     self.children.append(new_child)
#             #     new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
#             #         start_position=position,
#             #         direction=newDirectionChild,
#             #         vigor=exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
#             #         abundance=lerp(abundance,1,0.2), # children have more abundance
#             #         start_energy=distributed_energy
#             #     )

#             #     ringIdOfChildren.append(i)


#             #     direction = newDirectionParent
#             #     position += newDirectionParent * radius / 2 # divided by two so it does not stick out
#             # else:
#             #     # Dont split just go straight and turn upwards a lil
#             #     goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
#             #     goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
#             #     direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
#                 # abundance = lerp(abundance,0,0.05)# less abundance
#                 # vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
#             i += 1

#     def getTotalNumberOfBuds(self) -> int:
#         if len(self.bud_counts) == 0:
#             return 1 # every child is a bud rn
#         else:
#             return self.bud_counts[0][1] # first child, get count (not position)

#     def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
#         self.tube.compute_lengths()
#         obj.writeComment(f"Tube @ {at}")
#         obj.writeTube(self.tube, around_resolution, at=at, endConnectionMode=EndConnectionMode.CAPS)
#         for child in self.children:
#             obj.writeComment(f"Starting Child")
#             child.toObj(obj,at)


#     @staticmethod
#     def fromValue(value: float, direction: Vec3, seed: int) -> 'BetterBranch':
#         return BetterBranch(
#             abundance=  exerp(0.75,0.15,value),
#             seed=       seed,
#             direction=  direction,
#             vigor=      exerp(0.2,3.5,value),
#         )
    

#     def generateTubeDictData(self) -> List[List[dict]]:
#         result: List[List[dict]] = [self.tube.toDictList()]
#         if len(self.children) != 0:
#             for child in self.children:
#                 result += child.generateTubeDictData()
#         return result
            

#     def writeToDict(self) -> dict:
#         """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
#         return {
#             "tube": [
#                 self.generateTubeDictData()
#             ]
#         }





# # 3
# class BetterBranch():
#     children: List['BetterBranch'] # (if len 0 its a bud)
#     tube: Tube
#     bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

#     def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
#         position = start_position
#         self.children = []
#         ringIdOfChildren: List[int] = []
#         energy = start_energy
#         scale = 0.06 # multiplied by radii and positions as rings are created

#         rand = np.random.default_rng(seed)

#         def calc_radius() -> float:
#             # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
#             # +0.5 to prevent division by 0
#             vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
#             energy_to_radius_converter = 0.055
#             min_radius = 0.2 # because a branch a radius of 0 looks bad
#             return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
#         radius = calc_radius()

#         def calc_length() -> float:
#             # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
#             length_candidate = lerp(0.5,1,rand.random()) * vigor
#             min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
#             return min(length_candidate, min_length) 

#         self.tube = Tube([Ring(start_position*scale, radius*scale, direction.to_arbitrary_rotation())])

#         global BOBJ
#         global Boffset
#         global bigboy
#         if Boffset == 0:
#             bigboy = self
#         bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#         Boffset += 1

#         i = 0
#         while True:
#             radius = calc_radius()
#             length = calc_length()
#             position = (position + direction * length) # move along direction by length
#             self.tube.append(Ring(position*scale, radius*scale, direction.to_arbitrary_rotation()))
            
#             bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             Boffset += 1
            
#             energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
#             if energy < 0: # when its outta energy it finishes
#                 # End
#             #     self.tube.compute_lengths()
#             #     self.bud_counts = []
#             #     bud_count = 1 # every branch has a bud at the tip
#             #     for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
#             #         ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
#             #         ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
#             #         length = self.tube.lengths[ringIAfterSubDivide]
#             #         bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
#             #         self.bud_counts.append((length,bud_count)) 
#             #     self.bud_counts.reverse() # tip->base => base->tip
#             #     # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
#             #     # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
#             #     self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
#             #     self.tube.apply_parallel_transport() # make rings look at each other
#             #     self.tube = self.tube.subdivide() # make tube look smoother
#             #     self.tube.simplify_angles(0.05) # remove unnecessary geometry

#             #     bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             #     Boffset += 1

#                 return
#             if False and energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
#                 # Preform Split
#                 pass
#             #     splitAngle = rand.random() * math.tau
#             #     randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

#             #     newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
#             #     newDirectionParent = (direction - randomNormal).normalize()

#             #     energy -= radius # consumed expanding out
#             #     distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
#             #     energy -= distributed_energy


#             #     new_child = BetterBranch.__new__(BetterBranch)
#             #     self.children.append(new_child)
#             #     new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
#             #         start_position=position,
#             #         direction=newDirectionChild,
#             #         vigor=exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
#             #         abundance=lerp(abundance,1,0.2), # children have more abundance
#             #         start_energy=distributed_energy
#             #     )

#             #     ringIdOfChildren.append(i)


#             #     direction = newDirectionParent
#             #     position += newDirectionParent * radius / 2 # divided by two so it does not stick out
#             else:
#                 # Dont split just go straight and turn upwards a lil
#                 goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
#                 goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
#                 direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
#                 # abundance = lerp(abundance,0,0.05)# less abundance
#                 # vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
#             i += 1

#     def getTotalNumberOfBuds(self) -> int:
#         if len(self.bud_counts) == 0:
#             return 1 # every child is a bud rn
#         else:
#             return self.bud_counts[0][1] # first child, get count (not position)

#     def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
#         self.tube.compute_lengths()
#         obj.writeComment(f"Tube @ {at}")
#         obj.writeTube(self.tube, around_resolution, at=at, endConnectionMode=EndConnectionMode.CAPS)
#         for child in self.children:
#             obj.writeComment(f"Starting Child")
#             child.toObj(obj,at)


#     @staticmethod
#     def fromValue(value: float, direction: Vec3, seed: int) -> 'BetterBranch':
#         return BetterBranch(
#             abundance=  exerp(0.75,0.15,value),
#             seed=       seed,
#             direction=  direction,
#             vigor=      exerp(0.2,3.5,value),
#         )
    

#     def generateTubeDictData(self) -> List[List[dict]]:
#         result: List[List[dict]] = [self.tube.toDictList()]
#         if len(self.children) != 0:
#             for child in self.children:
#                 result += child.generateTubeDictData()
#         return result
            

#     def writeToDict(self) -> dict:
#         """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
#         return {
#             "tube": [
#                 self.generateTubeDictData()
#             ]
#         }




# # 4
# class BetterBranch():
#     children: List['BetterBranch'] # (if len 0 its a bud)
#     tube: Tube
#     bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

#     def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
#         position = start_position
#         self.children = []
#         ringIdOfChildren: List[int] = []
#         energy = start_energy
#         scale = 0.06 # multiplied by radii and positions as rings are created

#         rand = np.random.default_rng(seed)

#         def calc_radius() -> float:
#             # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
#             # +0.5 to prevent division by 0
#             vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
#             energy_to_radius_converter = 0.055
#             min_radius = 0.2 # because a branch a radius of 0 looks bad
#             return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
#         radius = calc_radius()

#         def calc_length() -> float:
#             # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
#             length_candidate = lerp(0.5,1,rand.random()) * vigor
#             min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
#             return min(length_candidate, min_length) 

#         self.tube = Tube([Ring(start_position*scale, radius*scale, direction.to_arbitrary_rotation())])

#         global BOBJ
#         global Boffset
#         global bigboy
#         if Boffset == 0:
#             bigboy = self
#         bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#         Boffset += 1

#         i = 0
#         while True:
#             radius = calc_radius()
#             length = calc_length()
#             position = (position + direction * length) # move along direction by length
#             self.tube.append(Ring(position*scale, radius*scale, direction.to_arbitrary_rotation()))
            
#             bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             Boffset += 1
            
#             energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
#             if energy < 0: # when its outta energy it finishes
#                 # End
#             #     self.tube.compute_lengths()
#             #     self.bud_counts = []
#             #     bud_count = 1 # every branch has a bud at the tip
#             #     for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
#             #         ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
#             #         ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
#             #         length = self.tube.lengths[ringIAfterSubDivide]
#             #         bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
#             #         self.bud_counts.append((length,bud_count)) 
#             #     self.bud_counts.reverse() # tip->base => base->tip
#             #     # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
#             #     # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
#             #     self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
#             #     self.tube.apply_parallel_transport() # make rings look at each other
#             #     self.tube = self.tube.subdivide() # make tube look smoother
#             #     self.tube.simplify_angles(0.05) # remove unnecessary geometry

#             #     bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             #     Boffset += 1

#                 return
#             if energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
#                 # Preform Split
#                 splitAngle = rand.random() * math.tau
#                 randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

#                 newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
#                 newDirectionParent = (direction - randomNormal).normalize()

#                 energy -= radius # consumed expanding out
#                 distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
#                 energy -= distributed_energy


#                 new_child = BetterBranch.__new__(BetterBranch)
#                 self.children.append(new_child)
#                 new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
#                     start_position=position,
#                     direction=newDirectionChild,
#                     vigor=vigor,#exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
#                     abundance=abundance,#lerp(abundance,1,0.2), # children have more abundance
#                     start_energy=distributed_energy
#                 )

#                 ringIdOfChildren.append(i)


#                 direction = newDirectionParent
#                 position += newDirectionParent * radius / 2 # divided by two so it does not stick out
#             else:
#                 # Dont split just go straight and turn upwards a lil
#                 goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
#                 goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
#                 direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
#                 # abundance = lerp(abundance,0,0.05)# less abundance
#                 # vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
#             i += 1

#     def getTotalNumberOfBuds(self) -> int:
#         if len(self.bud_counts) == 0:
#             return 1 # every child is a bud rn
#         else:
#             return self.bud_counts[0][1] # first child, get count (not position)


#     def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
#         self.tube.compute_lengths()
#         obj.writeComment(f"Tube @ {at}")
#         obj.writeTube(self.tube, around_resolution, at=at, endConnectionMode=EndConnectionMode.CAPS)
#         for child in self.children:
#             obj.writeComment(f"Starting Child")
#             child.toObj(obj,at)


#     @staticmethod
#     def fromValue(value: float, direction: Vec3, seed: int) -> 'BetterBranch':
#         return BetterBranch(
#             abundance=  exerp(0.75,0.15,value),
#             seed=       seed,
#             direction=  direction,
#             vigor=      exerp(0.2,3.5,value),
#         )
    

#     def generateTubeDictData(self) -> List[List[dict]]:
#         result: List[List[dict]] = [self.tube.toDictList()]
#         if len(self.children) != 0:
#             for child in self.children:
#                 result += child.generateTubeDictData()
#         return result
            

#     def writeToDict(self) -> dict:
#         """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
#         return {
#             "tube": [
#                 self.generateTubeDictData()
#             ]
#         }






# # 5
# class BetterBranch():
#     children: List['BetterBranch'] # (if len 0 its a bud)
#     tube: Tube
#     bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

#     def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
#         position = start_position
#         self.children = []
#         ringIdOfChildren: List[int] = []
#         energy = start_energy
#         scale = 0.06 # multiplied by radii and positions as rings are created

#         rand = np.random.default_rng(seed)

#         def calc_radius() -> float:
#             # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
#             # +0.5 to prevent division by 0
#             vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
#             energy_to_radius_converter = 0.055
#             min_radius = 0.2 # because a branch a radius of 0 looks bad
#             return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
#         radius = calc_radius()

#         def calc_length() -> float:
#             # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
#             length_candidate = lerp(0.5,1,rand.random()) * vigor
#             min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
#             return min(length_candidate, min_length) 

#         self.tube = Tube([Ring(start_position*scale, radius*scale, direction.to_arbitrary_rotation())])

#         global BOBJ
#         global Boffset
#         global bigboy
#         if Boffset == 0:
#             bigboy = self
#         bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#         Boffset += 1

#         i = 0
#         while True:
#             radius = calc_radius()
#             length = calc_length()
#             position = (position + direction * length) # move along direction by length
#             self.tube.append(Ring(position*scale, radius*scale, direction.to_arbitrary_rotation()))
            
#             bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             Boffset += 1
            
#             energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
#             if energy < 0: # when its outta energy it finishes
#                 # End
#             #     self.tube.compute_lengths()
#             #     self.bud_counts = []
#             #     bud_count = 1 # every branch has a bud at the tip
#             #     for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
#             #         ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
#             #         ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
#             #         length = self.tube.lengths[ringIAfterSubDivide]
#             #         bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
#             #         self.bud_counts.append((length,bud_count)) 
#             #     self.bud_counts.reverse() # tip->base => base->tip
#             #     # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
#             #     # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
#             #     self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
#             #     self.tube.apply_parallel_transport() # make rings look at each other
#             #     self.tube = self.tube.subdivide() # make tube look smoother
#             #     self.tube.simplify_angles(0.05) # remove unnecessary geometry

#             #     bigboy.toObj(BOBJ, Vec3(Boffset,0,0))
#             #     Boffset += 1

#                 return
#             if energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
#                 # Preform Split
#                 splitAngle = rand.random() * math.tau
#                 randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

#                 newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
#                 newDirectionParent = (direction - randomNormal).normalize()

#                 energy -= radius # consumed expanding out
#                 distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
#                 energy -= distributed_energy


#                 new_child = BetterBranch.__new__(BetterBranch)
#                 self.children.append(new_child)
#                 new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
#                     start_position=position,
#                     direction=newDirectionChild,
#                     vigor=exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
#                     abundance=lerp(abundance,1,0.2), # children have more abundance
#                     start_energy=distributed_energy
#                 )

#                 ringIdOfChildren.append(i)


#                 direction = newDirectionParent
#                 position += newDirectionParent * radius / 2 # divided by two so it does not stick out
#             else:
#                 # Dont split just go straight and turn upwards a lil
#                 goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
#                 goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
#                 direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
#                 abundance = lerp(abundance,0,0.05)# less abundance
#                 vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
#             i += 1

#     def getTotalNumberOfBuds(self) -> int:
#         if len(self.bud_counts) == 0:
#             return 1 # every child is a bud rn
#         else:
#             return self.bud_counts[0][1] # first child, get count (not position)

#     def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
#         self.tube.compute_lengths()
#         obj.writeComment(f"Tube @ {at}")
#         obj.writeTube(self.tube, around_resolution, at=at, endConnectionMode=EndConnectionMode.CAPS)
#         for child in self.children:
#             obj.writeComment(f"Starting Child")
#             child.toObj(obj,at)


#     @staticmethod
#     def fromValue(value: float, direction: Vec3, seed: int) -> 'BetterBranch':
#         return BetterBranch(
#             abundance=  exerp(0.75,0.15,value),
#             seed=       seed,
#             direction=  direction,
#             vigor=      exerp(0.2,3.5,value),
#         )
    

#     def generateTubeDictData(self) -> List[List[dict]]:
#         result: List[List[dict]] = [self.tube.toDictList()]
#         if len(self.children) != 0:
#             for child in self.children:
#                 result += child.generateTubeDictData()
#         return result
            

#     def writeToDict(self) -> dict:
#         """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
#         return {
#             "tube": [
#                 self.generateTubeDictData()
#             ]
#         }





# # 6
# class BetterBranch():
#     children: List['BetterBranch'] # (if len 0 its a bud)
#     tube: Tube
#     bud_counts: List[Tuple[float,int]] # ordered by first element which is position along. The second number is number of buds at that spot and before (implicit (infinity,1) should be at the end)

#     def __init__(self, seed, start_position: Vec3 = Vec3.zero(), direction: Vec3 = Vec3.X(), vigor: float = 0.5, abundance: float = 0.5, start_energy: float = 15):
#         position = start_position
#         self.children = []
#         ringIdOfChildren: List[int] = []
#         energy = start_energy
#         scale = 0.06 # multiplied by radii and positions as rings are created

#         rand = np.random.default_rng(seed)

#         def calc_radius() -> float:
#             # math.sqrt(vigor+0.5): sqrt because vigor is more about cross-sectional area and this make it radius based. Also vigor gets really big and really small, this confines its range a little bit more
#             # +0.5 to prevent division by 0
#             vigor_radius_factor = 1/(math.sqrt(vigor+0.5))
#             energy_to_radius_converter = 0.055
#             min_radius = 0.2 # because a branch a radius of 0 looks bad
#             return energy * energy_to_radius_converter * vigor_radius_factor + min_radius
        
#         radius = calc_radius()

#         def calc_length() -> float:
#             # * vigor because more vigorous means it should do less turning and less splitting - so it grows the same amount (because of how energy usage is calculated) but has less opertunities to do the other things
#             length_candidate = lerp(0.5,1,rand.random()) * vigor
#             min_length = energy / radius # to make it not go too far beyond when it gets to negative energy
#             return min(length_candidate, min_length) 

#         self.tube = Tube([Ring(start_position*scale, radius*scale, direction.to_arbitrary_rotation())])


#         i = 0
#         while True:
#             radius = calc_radius()
#             length = calc_length()
#             position = (position + direction * length) # move along direction by length
#             self.tube.append(Ring(position*scale, radius*scale, direction.to_arbitrary_rotation()))
            
            
#             energy -= length * radius + 0.01 # energy is based on surface area kinda? Being vigorous makes it thinner so it uses less energy and can go longer. +0.01 so it end eventually
#             if energy < 0: # when its outta energy it finishes
#                 # End
#                 self.tube.compute_lengths()
#                 # self.bud_counts = []
#                 # bud_count = 1 # every branch has a bud at the tip
#                 # for indexOfChild in range(len(ringIdOfChildren) - 1, -1, -1): # reverse order base->tip => tip->base (because we are counting bud_counts from the base to the tip)
#                 #     ringIBeforeSubdivide = ringIdOfChildren[indexOfChild]
#                 #     ringIAfterSubDivide = ringIBeforeSubdivide # the index of a ring doubles after one subdivision application
#                 #     length = self.tube.lengths[ringIAfterSubDivide]
#                 #     bud_count += self.children[indexOfChild].getTotalNumberOfBuds()
#                 #     self.bud_counts.append((length,bud_count)) 
#                 # self.bud_counts.reverse() # tip->base => base->tip
#                 # # get rid of unnecessary geometry (any ring that causes less than a 0.03radian curve in tube is discarded)
#                 # # last step because it doesn't change length calculations much but it does change ring indices unpredictably (which is required for the step above)
#                 # self.tube.simplify_distances() # remove rings next to each other that can form ugly sharp corners
#                 # self.tube.apply_parallel_transport() # make rings look at each other
#                 # self.tube = self.tube.subdivide() # make tube look smoother
#                 # self.tube.simplify_angles(0.05) # remove unnecessary geometry


#                 return
#             if energy > (1 + radius) and rand.random() < 0.3 and rand.random() < abundance: # energy > (1 + radius) to prevent it from splitting if it does not have enough energy; rand.random() < 0.3 to reduce the amount of splitting; rand.random() < abundance more abundance more splitting
#                 # Preform Split
#                 splitAngle = rand.random() * math.tau
#                 randomNormal = direction.to_arbitrary_perpendicular().rotate_around(direction,splitAngle)

#                 newDirectionChild = (direction + randomNormal).normalize() # they are 45deg from the current directions and 90 degrees from each other
#                 newDirectionParent = (direction - randomNormal).normalize()

#                 energy -= radius # consumed expanding out
#                 distributed_energy = (energy * rand.random() * 0.25) + 0.25 # child takes 1-26% of energy ~= 25 but 25-50% when energy ~= 1 (it is assumed that energy >= 1 at this location)
#                 energy -= distributed_energy


#                 new_child = BetterBranch.__new__(BetterBranch)
#                 self.children.append(new_child)
#                 new_child.__init__(seed= rand.integers(2**31-1), # just a random int to seed
#                     start_position=position,
#                     direction=newDirectionChild,
#                     vigor=exerp(vigor,3.5,0.5), # children are more vigorous (so they have make something with their life)
#                     abundance=lerp(abundance,1,0.2), # children have more abundance
#                     start_energy=distributed_energy
#                 )

#                 ringIdOfChildren.append(i)


#                 direction = newDirectionParent
#                 position += newDirectionParent * radius / 2 # divided by two so it does not stick out
#             else:
#                 # Dont split just go straight and turn upwards a lil
#                 goalDirection = (Vec3.random(rand) + Vec3.Y()).normalize() # mostly random but kinda upwards. Random because if its facing downwards, telling it to look up wont get it anywhere, it need to go to the side first
#                 goalPull = (1-1/(1+vigor/10))*0.7 # this curve was determined via experimentation in desmos ((0,0)(10,0.35)(infinity,0.7))
#                 direction = direction.lerp(goalDirection,goalPull) # look a lil more towards goal
#                 abundance = lerp(abundance,0,0.05)# less abundance
#                 vigor *= pow(1.1,direction.dot(Vec3.Y()))*0.9+0.1 # more vigor when its looking up (focus on getting tall); less when looking down (focus on facing down)
#             i += 1

#     def getTotalNumberOfBuds(self) -> int:
#         if len(self.bud_counts) == 0:
#             return 1 # every child is a bud rn
#         else:
#             return self.bud_counts[0][1] # first child, get count (not position)

#     def toObj(self, obj: ObjWriter, at: Vec3 = Vec3.zero(), around_resolution: int = 6) -> None:
#         if len(self.tube) != 1:
#             obj.writeComment(f"Tube @ {at}")
#             obj.writeTube(self.tube, around_resolution, at=at)
#         obj.writeComment(f"Cube @ {self.tube[-1].center + at}")
#         obj.writeCube(self.tube[-1].radius*2,self.tube[-1].center + at)
#         for child in self.children:
#             obj.writeComment(f"Starting Child")
#             child.toObj(obj,at)


#     @staticmethod
#     def fromValue(value: float, direction: Vec3, seed: int) -> 'BetterBranch':
#         return BetterBranch(
#             abundance=  exerp(0.75,0.15,value),
#             seed=       seed,
#             direction=  direction,
#             vigor=      exerp(0.2,3.5,value),
#         )
    

#     def generateTubeDictData(self) -> List[List[dict]]:
#         result: List[List[dict]] = [self.tube.toDictList()]
#         if len(self.children) != 0:
#             for child in self.children:
#                 result += child.generateTubeDictData()
#         return result
            

#     def writeToDict(self) -> dict:
#         """Make a dictionary with all the valuable metadata about the branch and its settings used to generate it."""
#         return {
#             "tube": [
#                 self.generateTubeDictData()
#             ]
#         }






            


if __name__ == '__main__':
    # print(Tube.from_verts([Vec3(0,0,0),Vec3(0,1,0),Vec3(0,2,1),Vec3(1,2,2),Vec3(2,1,2),Vec3(2,0,1),Vec3(2,0,1),Vec3(1,0,0)],0.5))
    with open("output.obj", "w") as fp:
        obj = ObjWriter(fp, "Output", False)



        position = 0


        upRot = Vec3.Y().to_arbitrary_rotation()
        tube = Tube([
            Ring(Vec3(0,-10,0),0.21,upRot),
            Ring(Vec3(0,0.1,0),0.21,upRot),
            Ring(Vec3(0,0.13,0),0.18,upRot),
            Ring(Vec3(0,0.15,0),0.15,upRot),
            Ring(Vec3(0,0.18,0),0.1,upRot),
            Ring(Vec3(0,0.2,0),0,upRot)
        ])

        for i in range(120):
            t = i/120
            for j in range(6):
                center = Vec3(position,0,0)
                obj.writeTube(tube,9)
                direction = Vec3.fromRotation(Rotation.from_rotvec([0,lerp(0,math.tau,j/6),0]))
                BetterBranch.fromValue(t, direction, j, 15).toObj(obj, center + direction*0.2)
            position += 5
            tube.translate(Vec3.X(5))
            



        # position = 0
        # energy = 30
        # v = 0.8
        # direction = Vec3(0,0,1)
        # seed = 50

        # def gen():
        #     global position, seed, v, direction
        #     BetterBranch.fromValue(v, direction, seed, energy).toObj(obj, Vec3(position,0,0))
        #     position += 10

        # for i in range(15):
        #     gen()
        #     v -= 0.02
        #     energy -= 1

        # for i in range(25):
        #     t = i/24
        #     gen()
        #     direction = -Vec3.fromRotation(Rotation.from_rotvec([0,lerp(math.tau/4,math.tau/2,t),0]))
        #     v -= 0.02
        
        # for i in range(50):
        #     t = i/49
        #     gen()
        #     direction = -Vec3.fromRotation(Rotation.from_rotvec([0,lerp(math.tau/2,math.tau,t),0]))
        #     seed = i+1
        #     v += 0.01

        # for i in range(10):
        #     t = i/9
        #     gen()
        #     direction = -Vec3.fromRotation(Rotation.from_rotvec([0,0,lerp(0,math.tau/4,t)]))

        # down = Rotation.from_rotvec([0,0,math.tau/4*3])

        # for i in range(10):
        #     t = i/9
        #     gen()
        #     direction = Vec3.fromRotation(Rotation.from_rotvec([lerp(0,-math.tau/4,t),0,0]) * down)
        #     energy += 1.5

        # BOBJ = obj
        # BetterBranch(
        #     abundance=  0.22,
        #     seed=       50,
        #     direction=  Vec3(0,0,1),
        #     vigor=      3,
        #     start_energy=30
        # )#.toObj(obj)
        print(Boffset)
        # obj.writeCube(2)


        # for i in range (10):
        #     BetterBranch.fromValue(i/9.0, Vec3(0,1,0),0).toObj(obj,Vec3(i,0,0),6)

        # ParametricBranch.from_vigor_spacing(seed=50).writeToObj(obj, Vec3(0,0,-0.5))
        # IterativeBranch(seed=60).writeToObj(obj,Vec3(0,0,0))
        # BetterBranch.fromValue(0.8,Vec3(1,0,0),50).toObj(obj, Vec3(0,0,0.5),12)
        

        # tube = Tube([
        #     Ring(center=Vec3(0,1,0), radius=0.5),
        #     Ring(center=Vec3(1,2,0), radius=0.5),
        #     Ring(center=Vec3(2,2,1), radius=0.5),
        #     Ring(center=Vec3(2,1,2), radius=0.5),
        #     Ring(center=Vec3(1,0,2), radius=0.5),
        #     Ring(center=Vec3(0,0,1), radius=0.5),
        #     Ring(center=Vec3(0,1,0), radius=0.5)
        # ])

        # tube.apply_parallel_transport()

        # obj.writeTube(tube=tube, vertsAround=6)










        # def update_directions(tube: Tube) -> None:
        #     for ring_index, ring in enumerate(tube):
        #         previous_ring_center = tube[ring_index-1].center if ring_index >= 1 else None
        #         next_ring_center = tube[ring_index+1].center if ring_index < len(tube) - 1 else None

        #         ring.rotation = (ring.center.get_tangent_optional_inputs(previous_ring_center, next_ring_center)).to_arbitrary_rotation()


        # tube = Tube([
        #     Ring(center=Vec3(0,1,0), radius=0.5),
        #     Ring(center=Vec3(1,2,0), radius=0.5),
        #     Ring(center=Vec3(2,2,1), radius=0.5),
        #     Ring(center=Vec3(2,1,2), radius=0.5),
        #     Ring(center=Vec3(1,0,2), radius=0.5),
        #     Ring(center=Vec3(0,0,1), radius=0.5),
        #     Ring(center=Vec3(0,1,0), radius=0.5)
        # ])

        # update_directions(tube)

        # obj.writeTube(tube=tube, vertsAround=6)


        # generate the thing for the blender animation 
        # count = 0
        # secondaryBranch = Tube.from_verts([Vec3.X(-30),Vec3.X(3630)],2)
        # obj.writeTube(secondaryBranch, 18)
        # for seed in range(3):
        #     for along in range(120):
        #         value = abs(1-along/60)
        #         rotation = along/120 + seed/3
        #         direction = Vec3.X().to_arbitrary_perpendicular().rotate_around(Vec3.X(),math.tau*rotation)
        #         x = seed * 5 + along * 30
        #         BetterBranch.fromValue(value, direction, seed).toObj(obj, Vec3.X(x) + direction)
        #         count+=1
        #         print(count/3.6)


        # tube = Tube([])
        # for i in range(-200,200):
        #     tube.append(Ring(Vec3(i*i/10000,i*i*i/1000000,i*i*i*i/100000000), 2+i/200))
        # tube.simplify_distances()
        # tube.apply_parallel_transport()
        # tube = tube.subdivide().subdivide()
        # tube.simplify_angles()
        # obj.writeTube(tube,12)




        # tube = Tube.from_verts([Vec3(0,2,1),Vec3(1,2,2),Vec3(2,1,2),Vec3(2,0,1),Vec3(1,0,0),Vec3(0,1,0)],0.1)
        # for ringdex, ring in enumerate(tube):
        #     ring.radius = lerp(0.05,0.25,ringdex/(len(tube)-1))

        # submissiveTube = tube.subdivide()
        # obj.writeTube(tube,8,at=Vec3(-2,0,-2))
        # obj.writeTube(submissiveTube,8,at=Vec3(2,0,2))


        # count = 0
        # for value in range(5):
        #     z = value * 15
        #     energy = 15
        #     vigor = exerp(0.1,2.5,value/4)
        #     abundance = lerp(0.75,0.25,value/4)
        #     secondaryBranch = Tube.from_verts([Vec3(0,0,z*1.5),Vec3(60*1.5,0,z*1.5)],2)
        #     obj.writeTube(secondaryBranch, 18)
        #     for i in range(0,60,4):
        #         direction = Vec3.X().to_arbitrary_perpendicular().rotate_around(Vec3.X(),lerp(0,math.tau,(i+60)/120+(i+120)/20))
        #         BetterBranch(
        #             abundance=abundance,
        #             vigor=vigor,
        #             energy_to_radius_converter=0.4/energy+0.028,
        #             start_energy=energy,
        #             seed=((i+248)%20)*43,
        #             direction=direction
        #         ).toObj(obj,Vec3(i*1.5,0,z*1.5) + 0.75 * direction)
        #         count+=1
        #         print(count/5/15*100)

        # obj.writeCube(0.4,Vec3(-1,-1,-1))
        # # print(obj.offset)
        # tube = Tube.from_verts([Vec3(0,1,0),Vec3(0,2,1),Vec3(1,2,2),Vec3(2,1,2),Vec3(2,0,1),Vec3(1,0,0)],0.5,True).subdivide()
        # obj.writeTube(tube,6)
        # spot = Vec3.zero()
        # spots = Ring(radius=10,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(4)
        # spots.extend(Ring(radius=20,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(8))
        # spots.extend(Ring(radius=30,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(12))
        # spots.extend(Ring(radius=40,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(16))
        # spots.extend(Ring(radius=50,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(20))
        # spots.extend(Ring(radius=60,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(24))
        # for index, spot in enumerate(spots):
        #     branch = BetterBranch(seed=index, direction=Vec3.random(random.randint(0,2**31-1)), energy_to_radius_converter = lerp(0.02,0.08,random.random()))
        #     branch.toObj(obj,spot)
        # # # print(branch.tube)
        #     print(index)
        

        # obj.writeComment("ring")
        # obj.writeSingleFace(Ring().convert_to_verts(16),Vec3(-0.5,0,0))
        # obj.writeComment("cube")
        # obj.writeCube(0.2,Vec3(0.5,0,0))


        # a = Tube([Ring(Vec3(0,-0.2,0)),Ring(Vec3(0,0,0)),Ring(Vec3(0,0.2,0)),Ring(Vec3(0,0.3,0.1)),Ring(Vec3(0,0.4,0.2)),Ring(Vec3(0.2,0.4,0.2))])
        # a.scale(0.5)
        # for ring in a:
        #     ring.radius /= 3
        # a.apply_parallel_transport()
        # a = a.subdivide().subdivide().subdivide().subdivide()
        # a.simplify_angles()
        # obj.writeComment("tube")
        # obj.writeTube(a,12, EndConnectionMode.CAPS)
        # obj.writeTube(a,12, EndConnectionMode.CAPS,Vec3(0,0,.1))
        # obj.writeTube(a,12, EndConnectionMode.CAPS,Vec3(0,0,.2))
        # obj.writeTube(a,12, EndConnectionMode.CAPS,Vec3(0,0,.3))
        # obj.writeTube(a,12, EndConnectionMode.CAPS,Vec3(0,0,.4))

        # spots = Ring(radius=3,rotation=Rotation.from_euler("xyz",[0,0,math.tau/4])).convert_to_verts(100)
        # for index, spot in enumerate(spots):
        #     IterativeBranch(seed=index,iterations=50,length=3, min_bud_spacing=10, max_bud_spacing=20).writeToObj(obj, spot)
        #     print(index)

        # for j in range(4,7):
        #     for i in range(4,7):
        #         IterativeBranch(seed=j,iterations=i+12,length=exerp(0.3,1.5,i/9)).writeToObj(obj, Vec3(i/2,0,j/2))

        # attempt 3:
        # go random direction and random curvature
        # moveforward random amount
        # get a couple options for new direction to go
        # choose option most correlated with goal
        # repeat


        # simulation constants
        # fStep = Vec3(0.12,0,0)
        # """How far each step of the simulation should go"""
        # rStep = Rotation.from_euler("xyz",[0,0.2,0])
        # """How much default curvature is added. Kinda simulates gravity but not really. Makes it generally more curvy and more interesting"""
        # maxMutation = math.tau/3
        # """The higher the number the larger the changes in direction will be"""
        # mutationPow = 5
        # """The higher the number the the less often large changes in direction will occur"""
        # goal = Vec3(0,1,0).normalize()
        # """Which direction the branch reaches towards"""
        # iterations = 16
        # """The number of simulation steps"""
        # subdivisionSteps: int
        # """How many time to subdivide before smoothing. Int >=0 dont go too high (>4) or it will be very slow for no benefit"""
        # smoothing: float
        # """in radians of minium allowed corner sharpness. Larger values are less smooth"""



                #     rand = np.random.default_rng(k)
                #     # rStep = Rotation.from_euler("xyz",[0,lerp(1.5,2.5,i/4),0])

                # # initial parameters
                #     ring = Ring(Vec3(0,0,0),0.1,Rotation.random(1,rand)[0])
                #     tube = Tube([ring])
                #     iterationRemaining = iterations
                #     while iterationRemaining > 0:
                #         iterationRemaining-=1

                #         rMutation = math.pow(rand.random(),mutationPow) * maxMutation

                #         # new possible directions
                #         optionA = ring.rotation * Vec3.random(rMutation, rand).to_euler_rotation() * rStep
                #         optionB = ring.rotation * Vec3.random(rMutation, rand).to_euler_rotation() * rStep
                #         optionC = ring.rotation * Vec3.random(rMutation, rand).to_euler_rotation() * rStep

                #         dotA = (Vec3(1,0,0) * optionA).dot(goal)
                #         dotB = (Vec3(1,0,0) * optionB).dot(goal)
                #         dotC = (Vec3(1,0,0) * optionC).dot(goal)

                #         choice: Rotation
                #         if dotA > dotB:
                #             if dotA > dotC:
                #                 choice = optionA
                #             else:
                #                 choice = optionC
                #         else:
                #             if dotB > dotC:
                #                 choice = optionB
                #             else:
                #                 choice = optionC
                        
                #         ring = Ring(ring.center + fStep * choice, lerp(0.05,0.1,iterationRemaining/iterations)*lerp(0.8,1,iterationRemaining/iterations), choice)
                #         tube.append(ring)

                #     # obj.writeTube(tube, 18, at=Vec3(-1,0,0))

                #     # make it beautiful
                #     tube.apply_parallel_transport()
                #     tube = tube.subdivide().subdivide().subdivide()
                #     tube.simplify_angles(0.02)

                #     obj.writeTube(tube, 150, at=Vec3(i,k*6,j))
                # print((j*4+k*20))

        









        # for seed in range(2,3):
        # i=0
        # for vigor in range(3,7):
        #     for spacing in range(3,7):
        #         i+=0.1
        #         Branch.from_vigor_spacing(vigor/9,spacing/9).writeToObj(obj, Vec3(0,0,i))
                #.write_all(f"output/{vigor}-{5}")



        # current_ring = Ring(Vec3(0,0,0), 0.1, Rotation.from_euler("xyz",[0,0,0]))
        # tube = Tube([current_ring])
        # step = 0.075
        # goal = Vec3(0,1,0)
        # rotationRatio = 1
        # spinFactor = 1.3
        # for _ in range(400):
        #     delta = step * Vec3(1,0,0) * current_ring.rotation * Rotation.from_euler("xyz",[0,0.5,0])
        #     rotationAxis = ((delta).cross(goal + delta)).normalize_or_zero().lerp(-(goal + delta).normalize_or_zero(), spinFactor)
        #     modificationRotation = Rotation.from_rotvec((rotationAxis * rotationRatio * step).np())
        #     current_ring = Ring(current_ring.center + delta, current_ring.radius, modificationRotation * current_ring.rotation)
        #     tube.append(current_ring)
        
        # a = Branch.from_vigor_spacing(5/9,9/9)
        # print(a.branch_tube.lengths)

        # tube.apply_parallel_transport()
        # a.writeToObj(obj, Vec3(0,0,0))
        # obj.writeTube(tube, 18, EndConnectionMode.CAP_BACK)
        # a.branch_tube.simplify_angles(0.03)
        # a.branch_tube = a.branch_tube.subdivide().subdivide()
        # a.branch_tube.simplify_angles(0.03)
        # a.along_resolution = len(a.branch_tube)
        # a.bud_locations = a.place_bud_locations()
        # a.writeToObj(obj, Vec3(0,0,2))
        # obj.writeTube(tube, 18, EndConnectionMode.CAP_BACK, Vec3(0,0,2))

    # with open("output.obj", "w") as fp:
    #     obj = ObjWriter(fp, True, "Output")

    #     for x in range(20):
    #         branch = Branch(bud_placement_seed=x, curve_seed=x)
    #         branch.writeToObj(obj, Vec3(x/4,x*PHI%1,0))
    #     writeToJson(branch.writeToDict(), "output.json")

        # delta_rot = Rotation.from_euler("xyz", [math.tau/2,0,math.tau/4])
        # for x in range(8):
        #     rot = Rotation.random()
        #     for y in range(8):
        #         obj.writeDirectionIndicator(1, rot, Vec3(x,y,0))
        #         rot = rot * delta_rot

        # obj.addComment("------Subdivision Test------")
        # rot_x = Rotation.identity()
        # rot_y = Rotation.from_euler("xyz",[0, 0, math.tau/4])
        # rot_z = Rotation.from_euler("xyz",[0, -math.tau/4, 0])
        # rot_neg_x = Rotation.from_euler("xyz",[math.tau/2, 0, math.tau/2])
        # # obj.addComment("===original===")
        # original = Tube([
        #     Ring(Vec3(0,0,0), 0.5, rot_y),
        #     Ring(Vec3(0,1,0), 0.5, rot_y),
        #     Ring(Vec3(1,2,0), 0.5, rot_x),
        #     Ring(Vec3(2,2,0), 0.5, rot_x),
        #     Ring(Vec3(3,2,1), 0.2, rot_z),
        #     Ring(Vec3(3,2,2), 0.2, rot_z),
        #     Ring(Vec3(2,2,3), 0.5, rot_neg_x)
        # ])
        # # obj.writeTube(original,128)
        # obj.addComment("===subdivision===")
        # obj.writeTube(original.subdivide(),16)

 

        # for i in range(0,30):
        #     t = i/30
        #     vertOffset = writeSingleFaceToObj(fp, vertOffset,
        #                                       Ring(Vec3(0,0,0), 0.5, TransformationMatrix.from_vector_roll(Vec3(1,1,1), lerp(0, math.tau, t))).convert_to_verts(16),
        #                                       Vec3(lerp(-15, 15, t),0,0)
        #                                     )



        # ring = Ring(Vec3(0,0,0), 0.5, TransformationMatrix.from_vector_rotation(Vec3(1,1,1),math.tau/4)).convert_to_verts(16)
        # for point in ring:
        #     vertOffset = writeCubeToObj(fp, vertOffset, 0.05, point)

        # tubeData = []
        # num_of_rings = 48
        # for i in range(num_of_rings):
        #     transformation = TransformationMatrix.from_pitch_yaw_roll(math.tau*i/num_of_rings, math.tau*i/num_of_rings/4, math.pi)
        #     # transformation = TransformationMatrix.from_vector_rotation(Vec3(1,1,1),0)
        #     tubeData.append(Ring(Vec3(0,i,0), 0.5, transformation))
        
        # vertOffset = writeTubeToObj(fp, vertOffset, create_tube(tubeData, 16), EndConnectionMode.NONE, Vec3(0,0,0))


        # obj.writeCube(1,Vec3(-3,0,-3))



        # # donuts - pretty much a golden path test. If this works then 90% of the stuff works
        # for f in range(9): # 9/36*math.tau = is 1/4 rotation
        #     for d in range(9):
        #         fp.write(f"# Donut {f}-{d}\n")
        #         transformation = Rotation.from_euler("xyz",[f/36*math.tau, d/36*math.tau, 0])
        #         num_of_rings = 18
        #         ring_size = 0.3
        #         direction = Rotation.from_euler("xyz",[0, f/36*math.tau, d/36*math.tau])
        #         ring = Ring(Vec3(0,0,0), ring_size, direction)
        #         donut_location = Vec3(d,f,5)

        #         obj.writeDirectionIndicator(ring_size * 2, direction, donut_location + Vec3(0,0,-2))
        #         # obj.writeSingleFace(ring, indicator_location)
        #         # obj.writeCube(ring_size/5, ring_size * Vec3(0.16,0.9,0) * direction + indicator_location)
        #         # obj.writeCube(ring_size/10, ring_size * Vec3(0.08,0.6,0) * direction + indicator_location)

        #         tube = Tube.from_verts(ring.convert_to_verts(num_of_rings), 0.1, True).subdivide()
        #         obj.writeTube(tube, 18, EndConnectionMode.CONNECTED, donut_location)