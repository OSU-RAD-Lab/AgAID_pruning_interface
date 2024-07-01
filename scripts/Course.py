
import sys
from typing import Optional
sys.path.append('../')
from scripts.Learning import LearningContent, ListLearningStructure , _LearningStructure, ProgressMarker, QuizStructure, RandomlyOrderedStringMultipleChoice
from enum import Enum

class QuizMode(Enum):
    AT_THE_END = False
    BUILD_OFF = True

class ModuleOrder(Enum):
    SPATIAL = False
    RULE = True


class Course(ListLearningStructure):
    _quizMode: QuizMode
    _moduleOrder: ModuleOrder
    introductionModule      :_LearningStructure
    introductionQuiz        :QuizStructure
    spacingModule           :_LearningStructure
    vigorModule             :_LearningStructure
    canopyModule            :_LearningStructure
    spacingQuiz             :QuizStructure
    vigorQuiz               :QuizStructure
    spacingVigorQuiz        :QuizStructure
    vigorCanopyQuiz         :QuizStructure
    spacingVigorCanopyQuiz  :QuizStructure
    vigorCanopySpacingQuiz  :QuizStructure
    spacingVigorCanopyFinal :_LearningStructure
    vigorCanopySpacingFinal :_LearningStructure

    def __init__(self, quizMode = QuizMode.BUILD_OFF, moduleOrder = ModuleOrder.RULE):
        super().__init__("Course")
        self._quizMode = quizMode
        self._moduleOrder = moduleOrder
        self.lengthDirty = True

        self.introductionModule     = createIntroductionModule()
        self.introductionQuiz       = createIntroductionQuiz()
        self.spacingModule          = createSpacingModule()
        self.vigorModule            = createVigorModule()
        self.canopyModule           = createCanopyModule()
        self.spacingQuiz            = createSpacingQuiz()
        self.vigorQuiz              = createVigorQuiz()
        self.spacingVigorQuiz       = createSpacingVigorQuiz()
        self.vigorCanopyQuiz        = createVigorCanopyQuiz()
        self.spacingVigorCanopyQuiz = createSpacingVigorCanopyQuiz()
        self.vigorCanopySpacingQuiz = createVigorCanopySpacingQuiz()
        self.spacingVigorCanopyFinal = createSpacingVigorCanopyFinal()
        self.vigorCanopySpacingFinal = createVigorCanopySpacingFinal()

        self.introductionModule     .parent = self
        self.introductionQuiz       .parent = self
        self.spacingModule          .parent = self
        self.vigorModule            .parent = self
        self.canopyModule           .parent = self
        self.spacingQuiz            .parent = self
        self.vigorQuiz              .parent = self
        self.spacingVigorQuiz       .parent = self
        self.vigorCanopyQuiz        .parent = self
        self.spacingVigorCanopyQuiz .parent = self
        self.vigorCanopySpacingQuiz .parent = self
        self.spacingVigorCanopyFinal.parent = self
        self.vigorCanopySpacingFinal.parent = self

        self.updateOrder()
    
    @property
    def quizMode(self) -> QuizMode:
        return self._quizMode

    @quizMode.setter
    def quizMode(self, value: QuizMode):
        self._quizMode = value
        self.updateOrder()

    @property
    def moduleOrder(self) -> ModuleOrder:
        return self._moduleOrder

    @moduleOrder.setter
    def moduleOrder(self, value: ModuleOrder):
        self._moduleOrder = value
        self.updateOrder()

    def addChild(self, child):
        raise TypeError("Course can not have children added to it with the addChildren method")

    def updateOrder(self) -> None:
        match (self.quizMode, self.moduleOrder) :
            case (QuizMode.BUILD_OFF, ModuleOrder.SPATIAL):
                self.children = [
                    self.introductionModule,
                    self.introductionQuiz,
                    self.spacingModule,
                    self.spacingQuiz,
                    self.vigorModule,
                    self.spacingVigorQuiz,
                    self.canopyModule,
                    self.spacingVigorCanopyQuiz,
                    self.vigorCanopySpacingFinal
                ]
            case (QuizMode.AT_THE_END, ModuleOrder.SPATIAL):
                self.children = [
                    self.introductionModule,
                    self.introductionQuiz,
                    self.spacingModule,
                    self.vigorModule,
                    self.canopyModule,
                    self.spacingVigorCanopyQuiz,
                    self.vigorCanopySpacingFinal
                ]
            case (QuizMode.BUILD_OFF, ModuleOrder.RULE):
                self.children = [
                    self.introductionModule,
                    self.introductionQuiz,
                    self.vigorModule,
                    self.vigorQuiz,
                    self.canopyModule,
                    self.vigorCanopyQuiz,
                    self.spacingModule,
                    self.vigorCanopySpacingQuiz,
                    self.spacingVigorCanopyFinal
                ]
            case (QuizMode.AT_THE_END, ModuleOrder.RULE):
                self.children = [
                    self.introductionModule,
                    self.introductionQuiz,
                    self.vigorModule,
                    self.canopyModule,
                    self.spacingModule,
                    self.vigorCanopySpacingQuiz,
                    self.spacingVigorCanopyFinal
                ]
        self.lengthDirty = True


def createIntroductionModule():
    root = ListLearningStructure("Introduction Module", ProgressMarker("Introduction to the course","start.png"))
    whatIsPruning = ListLearningStructure("What is Pruning?")
    root.addChild(whatIsPruning)
    whatIsPruning.addChild(LearningContent("Definition", "Pruning: The physical process of cutting back or removing unproductive branches on a tree."))
    whatIsPruning.addChild(LearningContent("Purpose", "Pruning is performed to maximize fruit quality vs quantity and maintain fruit growth stability over time."))
    whatIsPruning.addChild(LearningContent("Considerations",
"""There are three considerations that influence pruning:
1. Environment Management: Ensuring light and wind can be evenly distributed throughout the tree
2. Spacing of Fruit: Ensuring enough space for fruit to grow to quality we want 
3. Bud Counts: Maintain a number of buds in a given area of a tree"""))
    partsOfTree = ListLearningStructure("Parts of a Tree")
    root.addChild(partsOfTree)
    partsOfTree.addChild(LearningContent("Trunk","Trunk is the section of the tree that grows out of the ground"))
    partsOfTree.addChild(LearningContent("Secondary Branch","Secondary branches are tied to a wire for support"))
    partsOfTree.addChild(LearningContent("Tertiary Branch","Tertiary branches produce fruit\nThese are what you will be pruning"))
    partsOfTree.addChild(LearningContent("Bud","Buds are cone shaped objects that will produce the apples in the spring"))
    return root

def createIntroductionQuiz():
    root = QuizStructure("Introduction Quiz")
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which arrow points to a \"bud\"?",
        "this one",
        ["not this one", "nope", "you'd be silly to choose this one", "wrong answer"],
        "Find the Bud"
    ))
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which part of the tree is prunned?",
        "Tertiary Branches",
        ["Buds", "Secondary Branch", "Trunk"],
        "Prune Part"
    ))
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "This is a bonus question!\n Will RandomlyOrderedStringMultipleChoice be actually used in the interface when completed?",
        "No",
        ["Yes"],
        "Bonus Question"
    ))
    return root

def createSpacingModule():
    root = ListLearningStructure("Spacing Module")
    root.addChild(LearningContent("Spacing Module", "Spacing Module"))
    return root

def createVigorModule():
    root = ListLearningStructure("Vigor Module")
    root.addChild(LearningContent("Vigor Module", "Vigor Module"))
    return root

def createCanopyModule():
    root = ListLearningStructure("Canopy Module")
    root.addChild(LearningContent("Canopy Module", "Canopy Module"))
    return root

def createSpacingQuiz(title = "Spacing Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("test", "test.png")):
    root = QuizStructure(title, progressMarker)
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which quiz is this?",
        "Spacing",
        ["Vigor", "Canopy", "SpacingVigor", "VigorCanopy", "None of these options"],
        "Spacing Quiz"
    ))
    return root

def createVigorQuiz(title = "Vigor Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("test", "test.png")):
    root = QuizStructure(title, progressMarker)
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which quiz is this?",
        "Vigor",
        ["Spacing", "Canopy", "SpacingVigor", "VigorCanopy", "None of these options"],
        "Vigor Quiz"
    ))
    return root

def createSpacingVigorQuiz(title = "Vigor+ Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("test", "test.png")):
    root = QuizStructure(title, progressMarker)
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which quiz is this?",
        "SpacingVigor",
        ["Spacing", "Canopy", "Vigor", "VigorCanopy", "None of these options"],
        "SpacingVigor Quiz"
    ))
    return root

def createVigorCanopyQuiz(title = "Canopy+ Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("test", "test.png")):
    root = QuizStructure(title, progressMarker)
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which quiz is this?",
        "VigorCanopy",
        ["Spacing", "Canopy", "Vigor", "SpacingVigor", "None of these options"],
        "VigorCanopy Quiz"
    ))
    return root

def createSpacingVigorCanopyQuiz(title = "Canopy++ Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("test", "test.png")):
    root = QuizStructure(title, progressMarker)
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which quiz is this?",
        "None of these options",
        ["Spacing", "Canopy", "Vigor", "SpacingVigor", "VigorCanopy"],
        "SpacingVigorCanopy Quiz"
    ))
    return root

def createVigorCanopySpacingQuiz(title = "Spacing++ Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("test", "test.png")):
    root = QuizStructure(title, progressMarker)
    root.addChild(RandomlyOrderedStringMultipleChoice(
        "Which quiz is this?",
        "None of these options",
        ["Spacing", "Canopy", "Vigor", "SpacingVigor", "VigorCanopy"],
        "VigorCanopySpacing Quiz"
    ))
    return root

def createVigorCanopySpacingFinal():
    root = ListLearningStructure("Final Test", ProgressMarker("Show you knowledge!", "final.png"))
    root.addChild(createVigorQuiz("", None))
    root.addChild(createVigorCanopyQuiz("", None))
    root.addChild(createVigorCanopySpacingQuiz("", None))
    return root

def createSpacingVigorCanopyFinal():
    root = ListLearningStructure("Final Test", ProgressMarker("Show you knowledge!", "final.png"))
    root.addChild(createSpacingQuiz("Easy", None))
    root.addChild(createSpacingVigorQuiz("Medium", None))
    root.addChild(createSpacingVigorCanopyQuiz("Hard", None))
    return root