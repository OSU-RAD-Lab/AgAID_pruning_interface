# hierarchal/OOP data structure to represent the entire learning course as a tree
# In the tree, the vertices are subclasses of _LearningComponent
# Something is is viewable and requires pressing the NEXT button to continue if it has self.viewable = True
# If something has a title and is an ancestor of the currently viewed, then there will be a progress bar for it.
# _LearningStructure (and subclasses of it) are currently the only _LearningComponent that have children
# You may wish to subclass _LearningStructure if you want to be able to dynamically change the ordering of the children (for example top level Course to change the order of modules)
# _LearningStructure is currently the only class that can take ProgressMarkers which is not a _LearningComponent but will show up with an image and a description
# QuizStructure is a subclass of _LearningStructure that is designed to have _QuizQuestion as children
# _QuizQuestion is a class that should only be subclassed but will be able to automatically create new randomized instances of itself (if its a child of QuizStructure) when it is gotten wrong
# Calling next on a _LearningComponent should return the the next viewable _LearningComponent or None if it is at the end - due to the tree structure of the course, depth first search (with a pre-ordering) is implemented to do so

# classes that starts with _ such as _LearningComponent, _LearningStructure, and _QuizQuestion should not be instantiated and should only be subclassed

from enum import Enum
import random
import copy
from typing import Any, List, Optional, Tuple, Union

# Used for returns from when .getProgress() is called on _LearningComponent 
class ProgressReturn:
    completed: int
    total: int
    owner: '_LearningStructure'

    def __init__(self, completed: int, total:int, owner):
        self.completed = completed
        self.total = total
        self.owner = owner

    @property
    def title(self):
        return self.owner.title

    # return the amount the way through as a number between 0 and 1
    # bias is added on so a bias of 0 will count the current question as incomplete (you will never get to 100%) and a bias of 1 will count the current question as completed (you will start with some progress)
    # a bias of 0.5 will be in the middle (both downsides)
    def value(self, bias: float = 0) -> float:
        return (self.completed + bias) / self.total
    
    # just a nice way to see what a list of progress returns looks like
    @staticmethod
    def makeStringFromList(list: List['ProgressReturn'], bias = 0.5) -> str:
        result = ""
        for i in list[::-1]:
            result += f"{i.title} {int(100 * i.value(bias))}%\n"
        
        return result

# for handling a single component without any children without any content
# everything that is a learning component extends this

# should only be subclassed
# if you image a course as a tree, subclasses of this represent vertices
class _LearningComponent:
    title: str
    parent: Optional['_LearningStructure']
    viewable: bool

    # `viewable` should be true for any component that can be viewed
    def __init__(self, title = "", viewable = True):
        self.title = title
        self.parent = None # every component should only have zero or one parent. A component can have a parent that does not recognize it as a child as long the component is not directly interacted with.
        self.viewable = viewable

    # used to find the next viewable component in the course
    # that next component is returned
    # can return None if unable to find another viewable component in the rest of the course
    # traverses the course in depth-first search with pre-ordering
    def next(self) -> Optional['_LearningComponent']:
        if self.parent is None: return None
        return self.parent.intoNext(self)
    
    # same as next() but the other direction
    def prev(self) -> Optional['_LearningComponent']:
        if self.parent is None: return None
        return self.parent.intoPrev(self)
    
    # while next() goes to the next _LearningComponent, intoNext(previous) tries to enter this _LearningComponent - but if it is not viewable, it goes onto the next one
    # it returns a _LearningComponent (which might be itself or a different one or None)
    # if it is not viewable then it calls next to determine what the next component should be
    # `previous` is mandatory and should be the component that called this (via .intoNext(self)) (this is so that _LearningStructure and other subclasses can know where to continue the search from)
    # see next() for more information
    def intoNext(self, previous: '_LearningComponent') -> Optional['_LearningComponent']:
        if self.viewable: return self
        return self.next()

    # same as intoNext(previous) but the other direction
    def intoPrev(self, next: '_LearningComponent') -> Optional['_LearningComponent']:
        if self.viewable: return self
        return self.prev()

    # length is the total number of viewable LearningComponents in this LearningComponents subtree (this LearningComponents and all its descendants)
    # this is used for the progress bar - so a _LearningComponent with no children is 0 or 1
    def length(self) -> int:
        if (self.viewable): return 1
        else: return 0
    
    # gets parents till there are no more then returns that
    def getRoot(self) -> '_LearningComponent':
        if self.parent is None: return self
        return self.parent.getRoot()
    
    # get a list of all ancestors with a title captured in a progress return
    def getProgress(self) -> List[ProgressReturn]:
        if self.parent is None: return []
        return self.parent.getProgressFromChild(self, 0, [])
    
    # gets the nearest ancestral progress marker
    def getProgressMarker(self) -> Optional['ProgressMarker']:
        if self.parent is None: return None
        else: return self.parent.getProgressMarker()

    # returns true if recursively self.parent.hasChild(self) returns True - meaning that from the root it is possible to navigate to this Component
    def isConnectedToRoot(self, previous = None):
        if previous is not None: return False
        if not self.viewable: return False
        if self.parent is None: return True
        return self.parent.isConnectedToRoot(self)


# This will hold all the data that will be displayed on the page
# currently it just holds a string but that will change!
# In the future it should hold the information such as
#     which model to show, where to show it, how to orient the camera,
#     wether or not this is a quiz, if it is what the questions are,
#     any call-outs where those are what they say,
#     what controls the user has etc
class Content():
    content: str

    def __init__(self, content = ""):
        self.content = content


# has a view method and can return some content
class _LearningContent():

    # should return in the form of a 2 sized tuple, the new thing to LearningComponent and the actual content to display 
    def view(self) -> Tuple[_LearningComponent, Content]:
        return (self, None) # type: ignore
    

# if a learning structure has this it will show up on the task view
class ProgressMarker:
    description: str
    image: str
    # description and image are both mandatory strings
    # image should be a path to an image to display
    def __init__(self, description: str, image: str):
        self.description = description
        self.image = image

# The _LearningComponent that can have children
# this should only be subclassed a simple implementation of this is StaticLearningStructure
# if doing so only the last couple methods need to be modified and then it should be able to handle changes in ordering and length
# NOTICE: if the length is changed make sure to set self.lengthDirty = True (if addChild is used it is handled for you)
# this is because the length (total number of viewable descendants) is cached as it might be mildly expensive and might need to be repeated read 
class _LearningStructure(_LearningComponent):
    progressMarker: Optional[ProgressMarker]
    lengthDirty: bool
    cachedLength: int
    # `title` is optional, and it will always be displayed in the in the decedent of it is being viewed
    # `progressMarker` is optional should be a ProgressMarker
    # progressMarker will only be displayed when a descendant is being viewed that does not have a more recent parent that does with progressMarker
    def __init__(self, title = "", progressMarker = None):
        super().__init__(title, False)
        self.progressMarker = progressMarker
        self.lengthDirty = False
        self.cachedLength = 0
    
    # children is an list of LearningComponents which are all added as children to this _LearningStructure
    # each child must follow the rules defined by .addChild(child)
    def addChildren(self, children: List[_LearningComponent]) -> None:
        for child in children:
            self.addChild(child)
    
    # returns the course progressMarker, if it does not have one it bubbles up till it gets till one which is returned or it reaches the root
    # can return None if this and all its ancestors do not have a ProgressMarker
    def getProgressMarker(self) -> Optional['ProgressMarker']:
        if self.progressMarker is not None: return self.progressMarker
        if self.parent is None: return None
        else: return self.parent.getProgressMarker()
        
    # same as superclass's but implemented to handle having children
    # calling this on this _LearningStructure means that the _LearningStructure has already been called so its time to go on to the next one
    # the next one would be its first child
    # so if there are children first considers the first child instead
    # if there are not children then it calls the superclass's next (which will raise to result in raising to its parent)
    def next(self) -> Optional[_LearningComponent]:
        firstChild = self.firstChild()
        if firstChild is None: return super().next()
        return firstChild.intoNext(self)
    
    # same as next() but in the opposite direction
    def prev(self) -> Optional[_LearningComponent]:
        lastChild = self.lastChild()
        if lastChild is None: return super().prev()
        return lastChild.intoPrev(self)
    
    # same as superclass's but implemented to handle having children
    def intoNext(self, previous: _LearningComponent) -> Optional[_LearningComponent]:
        # calling this from something that is not a child means is assumed to be the first visit - so because of pre-ordering call its super().intoNext(previous) to go into this _LearningComponent
        if not self.hasChild(previous): return super().intoNext(previous)
        # if it is a child, then check if its the last one - if so then its done with this subtree and will go to the next one (either a sibling or above in the tree) done by using super().next()
        if previous == self.lastChild(): return super().next()
        # otherwise it just does the next child after the current one (and the nex child does exist because its not the last one)
        return self.nextChild(previous).intoNext(self) # type: ignore
    
    # same as intoNext(self) but in reverse
    def intoPrev(self, next: _LearningComponent) -> Optional[_LearningComponent]:
        if not self.hasChild(next): return super().intoPrev(next)
        if next == self.firstChild(): return super().prev()
        result = self.prevChild(next)
        return result.intoPrev(self) # type: ignore

    # see superclass's comment for purpose
    # uses a cached length because self.recomputeLengthTo(self.lastChild()) is costly
    # if anything modifies the length it should self.lengthDirty = True so the next check of the length will take less time
    # will set its parent's length dirty if nessarary because its length changed
    def length(self) -> int:
        if self.lengthDirty:
            newLength = self.recomputeLengthTo(self.lastChild())
            if newLength != self.cachedLength:
                self.cachedLength = newLength
                if self.parent != None:
                        self.parent.lengthDirty = True
            self.lengthDirty = False

        return self.cachedLength
    
    def getProgressFromChild(self, child: _LearningComponent, current: int, array: List[ProgressReturn]) -> List[ProgressReturn]:
        if child == self.firstChild():
            progressTill = 0
        else:
            progressTill = self.recomputeLengthTo(self.prevChild(child))
        totalCompleted = progressTill + current

        if self.title != "":
            array.append(ProgressReturn(totalCompleted, self.length(), self))

        if self.parent is None: return array
        return self.parent.getProgressFromChild(self, totalCompleted, array)
    
    # find the total length to a given child (inclusive)
    # endChild must be a child of this _LearningStructure
    # because length is cached, this should typically not do much recursion (only when things are changed)
    def recomputeLengthTo(self, endChild: Optional[_LearningComponent]) -> int:
        if not self.hasChild(endChild): raise IndexError
        child = self.firstChild()
        if child is None: return 0
        length = 0
        for _ in range(100000):
            length += child.length() # type: ignore
            if child == endChild: return length
            child = self.nextChild(child) # type: ignore - child should never return None before == endChild
        raise OverflowError("Probable infinite loop: Was not able to find endChild by calling nextChild within 100000 times on firstChild")

    # returns true if recursively self.parent.hasChild(self) returns True - meaning that from the root it is possible to navigate to this Component
    def isConnectedToRoot(self, previous = None):
        if previous is not None: 
            if not self.hasChild(previous):
                return False
        if self.parent is None: return True
        return self.parent.isConnectedToRoot(self)


    # subclasses of _LearningStructure that modify the order of children / store children in a different way than an list  will need to modify these couple methods
    
    # !!you should call the this superclass as well!!
    # `child` should be a subclasses of _LearningComponent
    # it must not already have a parent
    def addChild(self, child: _LearningComponent) -> None:
        if not (child.parent is None):
            raise ValueError("Child already has a parent")
        child.parent = self
        self.lengthDirty = True

    # returns the last child in the current ordering
    # returns none if their are no children
    def lastChild(self) -> Optional[_LearningComponent]:
        return None

    # returns the first child in the current ordering
    # returns none if there are no children
    def firstChild(self) -> Optional[_LearningComponent]:
        return None

    # takes argument child and returns true if that is a child (DIRECT descendant)
    def hasChild(self, child: Optional[_LearningComponent]) -> bool:
        return False

    # takes argument previous and returns the next child after that one
    # assumes that `self.hasChild(previous) => True` and `previous != self.lastChild()`
    def nextChild(self, previous: _LearningComponent) -> Optional[_LearningComponent]:
        return None
    
    # takes argument next and returns the previous child to that one
    # assumes that `self.hasChild(next) => True` and `next != self.firstChild()`
    def prevChild(self, next: _LearningComponent) -> Optional[_LearningComponent]:
        return None

# a simple implementation of _LearningStructure using a list
class ListLearningStructure(_LearningStructure):
    children: List[_LearningComponent]

    def __init__(self, title = "", progressMarker = None):
        super().__init__(title, progressMarker)
        self.children = []
    
    # see _LearningStructure's implementation
    def addChild(self, child: _LearningComponent):
        self.children.append(child)
        super().addChild(child)
    
    # see _LearningStructure's implementation
    def lastChild(self) -> Optional[_LearningComponent]:
        if len(self.children) == 0: return None
        return self.children[len(self.children)-1]
    
    # see _LearningStructure's implementation
    def firstChild(self) -> Optional[_LearningComponent]:
        if len(self.children) == 0: return None
        return self.children[0]
    
    # see _LearningStructure's implementation
    def hasChild(self, child: Optional[_LearningComponent]) -> bool:
        return child in self.children
    
    # see _LearningStructure's implementation
    def nextChild(self, previous: _LearningComponent) -> Optional[_LearningComponent]:
        index = self.children.index(previous)
        return self.children[index+1]
    
    # see _LearningStructure's implementation
    def prevChild(self, next: _LearningComponent) -> Optional[_LearningComponent]:
        index = self.children.index(next)
        return self.children[index-1]


class LearningContent(_LearningComponent, _LearningContent):
    content: Content
    
    # this is a place holder and an example - this should be replaced with multiple other classes that have other real pieces of content
    # for example a page that displays text in a formatted way or a video that plays or a question or the result of a question - the possibilities are endless
    # for things that display, a title is recommended (to skip use empty string)
    def __init__(self, title: str, content: str):

        super().__init__(title)
        self.content = Content(content)

    def view(self) -> Tuple[_LearningComponent, Content]:
        return (self, self.content)
            
# pretends to be have a length larger even though it is actually only one instance
# need to call next multiple times to get through it
# !! !! !! WARNING !! !! !! not rigorously tested, and probably has multiple bugs
# however, it does the trick in its current use case as a subclass for UnviewedQuizQuestion
# # so I dont think spending development time on it right now is important
class _LearningMultiComponent(_LearningComponent):
    current: int
    _size: int

    def __init__(self, size: int, title = ""):
        super().__init__(title)
        self.current = 0
        self._size = size
        self._viewable = self.viewable
        self.size = size

    @property
    def size(self) -> int:
        return self._size
    
    # make it not viewable if the size is 0
    @size.setter
    def size(self, size: int):
        if size < 0:
            raise ValueError(f"Size of {self.__class__} can not be below zero")
        if size == self._size:
            return
        if isinstance(self.parent, _LearningStructure):
            self.parent.lengthDirty = True
        if size == 0:
            self._viewable = self.viewable
            self.viewable = False
        else: 
            self.viewable = self._viewable
        self._size = size

    # get the instance of this but it is at the beginning
    def first(self) -> '_LearningMultiComponent':
        self.current = 0
        return self
    
    # get the instance of this but it is at the end
    def last(self) -> '_LearningMultiComponent':
        self.current = self.size - 1
        return self

    # will return itself and increment its current
    # or will return None if it is at the size limit
    def nextPart(self) -> Optional['_LearningMultiComponent']:
        if self.current < self.size - 1:
            self.current += 1
            return self
        return None
    
    # same as nextPart but reverse
    def prevPart(self) -> Optional['_LearningMultiComponent']:
        if self.current > 0:
            self.current -= 1
            return self
        return None

    # see superclass
    def next(self) -> Optional[_LearningComponent]:
        next = self.nextPart()
        if not (next is None): return next
        if self.parent is None: return None
        return self.parent.intoNext(self)
    
    # see superclass
    def prev(self) -> Optional[_LearningComponent]:
        prev = self.prevPart()
        if not (prev is None): return prev
        if self.parent is None: return None
        return self.parent.intoNext(self)
    
    # see superclass
    def length(self) -> int:
        if (self.viewable): return self.size
        else: return 0

    # potentially broken
    # see superclass
    def getProgress(self) -> List[ProgressReturn]:
        if self.size <= 1 or self.title == "" or not self.viewable: return super().getProgress()
        if self.parent is None: return [ProgressReturn(self.current, self.length(), self)]
        return self.parent.getProgressFromChild(self, self.length(), [ProgressReturn(self.current, self.length(), self)])

# should not be created outside of being used within QuizStructure
# is used to make QuizStructure seem the appropriate length
class UnviewedQuizQuestion(_LearningMultiComponent, _LearningContent):
    parent: 'QuizStructure'

    def __init__(self):
        super().__init__(0, "???")
        self.viewable = True

    def view(self) -> Tuple[_LearningComponent, Content]:
        if self.size == 0:
            return (self, Content())
        return self.parent.viewUnviewed()

# This is a subclass of _LearningStructure that can only have subclasses of _QuizQuestion as children
# It will hold all its children in a undetermined order
# The order of a given child is determined when it is viewed
# When a question is gotten wrong a variant of that child is made to be asked again soon
# it has 3 internal lists for questions
#   viewedChildren - questions that have been viewed (their order will not change)
#   randomizeBucket - questions are randomly selected from here when an new question is to be viewed
#   holdingBucket - when a question was gotten wrong a new variant of that question is placed here. When randomizeBucket is emptied this is moved over to it. (This prevents randomly accidentally asking the same question over and over again while there are still more left)
# however when it is looked at from the interface:
#   it displays only the already viewed questions
#   then `unviewed` a multiple times equivalent to the number of questions remaining (elements in the randomizeBucket and the holdingBucket)
class QuizStructure(_LearningStructure):
    viewedChildren: List['_QuizQuestion']
    randomizeBucket: List['_QuizQuestion']
    holdingBucket: List['_QuizQuestion']
    unviewed: UnviewedQuizQuestion

    def __init__(self, title = "Quiz", progressMarker: Optional[ProgressMarker] = ProgressMarker("Quiz","quiz.png")):
        super().__init__(title, progressMarker)
        self.viewedChildren = []
        self.randomizeBucket = []
        self.holdingBucket = []
        self.unviewed = UnviewedQuizQuestion()
        self.unviewed.parent = self

    # adds a child to the randomizeBucket
    def addChild(self, child: '_QuizQuestion'):
        if not (child.parent is None):
            raise ValueError("Child already has a parent")
        if not isinstance(child, _QuizQuestion):
            raise TypeError("QuizStructure can only have _QuizQuestions as children")
        child.parent = self
        self.randomizeBucket.append(child)
        self.lengthDirty = True
        self.unviewed.size += 1

    # !! should not be used externally and should only be called by a _QuizQuestion when it is marked wrong !!
    # adds a newChild to holdingBucket
    def childIsWrong(self, newChild: '_QuizQuestion'):
        if not (newChild.parent is None):
            raise ValueError("newChild already has a parent")
        newChild.parent = self
        self.holdingBucket.append(newChild)
        self.lengthDirty = True
        self.unviewed.size += 1


    # !! should not be used externally and only be called by UnviewedQuizQuestion that lives in self.unviewed by calling self.parent.viewUnviewed()
    # instead of viewing the UnviewedQuizQuestion, it views a randomChild from randomizeBucket
    # transfers holdingBucket to randomizeBucket if nessarary
    # errors if both buckets are empty or the size of UnviewedQuizQuestion is zero
    def viewUnviewed(self) -> Tuple[_LearningComponent, Content]:
        if self.unviewed.size == 0:
            raise OverflowError("UnviewedQuizQuestion with size zero was attempted to be viewed - which is illegal")
        self.unviewed.size -= 1
        if len(self.randomizeBucket) == 0:
            if len(self.holdingBucket) == 0:
                raise OverflowError("viewUnviewed was called with BOTH BUCKETS EMPTY! - dont do that")
            self.randomizeBucket = self.holdingBucket
            self.holdingBucket = []
        randomIndex = random.randint(0, len(self.randomizeBucket) - 1)
        randomChild = self.randomizeBucket.pop(randomIndex)
        self.viewedChildren.append(randomChild)
        return randomChild.view()
    
    # implementation of _LearningStructure method, see that for more info
    def lastChild(self) -> Optional[_LearningComponent]: 
        if self.unviewed.size == 0:
            if len(self.viewedChildren) == 0:
                    return None
            else:
                return self.viewedChildren[len(self.viewedChildren)-1]
        else:
            return self.unviewed.last()
    
    # implementation of _LearningStructure method, see that for more info
    def firstChild(self) -> Optional[_LearningComponent]:
        if len(self.viewedChildren) == 0:
            if self.unviewed.size == 0:
                return None
            else:
                return self.unviewed.first()
        else:
            return self.viewedChildren[0]
        
    # see superclass
    def hasChild(self, child: Optional[_LearningComponent]) -> bool:
        return child in self.viewedChildren or child == self.unviewed

    # implementation of _LearningStructure method, see that for more info
    # there are 20 different states relevant variables can be in
    # 5 only 5 of those state are assumed to occur based on the rules that:
    # hasChild(previous) == true
    # lastChild(previous) != previous
    # those 5 plus 2 others are correctly handled
    # cases:
    # would it correctly respond | state code | full state description | state number
    #   XX-- not a child (ASSUMED TO NEVER OCCUR)
    #   --00 unviewed==0, viewed==0 (ASSUMED TO NEVER OCCUR)
    # x LU!! last child of unviewed, unviewed!=0, viewed!=0 (1) (ASSUMED TO NEVER OCCUR)
    # x fU!! first or mid child of unviewed, unviewed!=0, viewed!=0 (2*)
    # x Lv!! last child of viewed, unviewed!=0, viewed!=0 (3*)
    # x fv!! first or mid child of viewed, unviewed!=0, viewed!=0 (4*)
    #   -U0- child of unviewed, unviewed == 0 (ASSUMED TO NEVER OCCUR)
    # x LU!0 last child of unviewed, unviewed!=0, viewed==0 (5) (ASSUMED TO NEVER OCCUR)
    # x fU!0 first of mid child of unviewed, unviewed!=0, viewed==0 (6*)
    #   -v-0 child of viewed, viewed == 0 (ASSUMED TO NEVER OCCUR)
    #   Lv0! last child of viewed, unviewed==0, viewed!=0 (7) (ASSUMED TO NEVER OCCUR)
    # x fv0! first or mid child of viewed, unviewed==0, viewed!=0 (8*)
    def nextChild(self, previous: Union['_QuizQuestion', UnviewedQuizQuestion]) -> Optional[_LearningComponent]:
        if previous == self.unviewed: # -U-- assumed -U!-
            return previous.nextPart() # type: ignore # (1) & (2*) & (5) & (6*)
        else: # -v-- assumed -v-!
            index = self.viewedChildren.index(previous) # type: ignore # requires ---!
            if index == len(self.viewedChildren) - 1: # Lv-! assumed Lv!!
                return self.unviewed.first() # (3*)
            else: #fv-
                return self.viewedChildren[index+1] # (4*) & (8*)
    
    # implementation of _LearningStructure method, see that for more info
    # there are 20 different states relevant variables can be in
    # 5 only 5 of those state are assumed to occur based on the rules that:
    # hasChild(previous) == true
    # lastChild(previous) != previous
    # those 5 are correctly handled
    # cases:
    # would it correctly respond | state code | full state description | state number
    #   XX-- not a child (ASSUMED TO NEVER OCCUR)
    #   --00 unviewed==0, viewed==0 (ASSUMED TO NEVER OCCUR)
    #   -U0- child of unviewed, unviewed == 0 (ASSUMED TO NEVER OCCUR)
    #   -v-0 child of viewed, viewed == 0 (ASSUMED TO NEVER OCCUR)
    # x lU!! last or mid child of unviewed, unviewed!=0, viewed!=0 (1*)
    # x FU!! first child of unviewed, unviewed!=0, viewed!=0 (2*)
    # x lv!! last or mid child of viewed, unviewed!=0, viewed!=0 (3*)
    #   Fv!! first child of viewed, unviewed!=0, viewed!=0 (4) (ASSUMED TO NEVER OCCUR)
    # x lU!0 last or mid child of unviewed, unviewed!=0, viewed==0 (5*)
    #   FU!0 first child of unviewed, unviewed!=0, viewed==0 (6) (ASSUMED TO NEVER OCCUR)
    # x lv0! last or mid child of viewed, unviewed==0, viewed!=0 (7*)
    #   Fv0! first child of viewed, unviewed==0, viewed!=0 (8) (ASSUMED TO NEVER OCCUR)
    def prevChild(self, next: Union['_QuizQuestion', UnviewedQuizQuestion]) -> Optional[_LearningComponent]:
        if next == self.unviewed: # -U-- assumed -U!-
            prevPart = next.prevPart() # type: ignore # requires -U--
            if prevPart is None: # FU!- assumed FU!!
                return self.viewedChildren[len(self.viewedChildren) - 1] # (2*)
            else: # lU!-
                return self.unviewed.prevPart() # (1*) & (5*)
        else: # -v-- assumed lv-!
            index = self.viewedChildren.index(next) # type: ignore # requires -v-!
            return self.viewedChildren[index-1] # (3*) & (7*) requires lv-!
    
class QuestionState(Enum):
    INCOMPLETE = 0
    INCORRECT = 1
    CORRECT = 2

# another class that should only be a subclassed and not instanced itself
class _QuizQuestion(_LearningComponent, _LearningContent):
    state: QuestionState

    def __init__(self, title = ""):
        super().__init__(title)
        self.viewable = True
        self.state = QuestionState.INCOMPLETE # the state is managed by the superclass - only read it (and dont write to it) in a subclass please

    # answering the question counts as one and moving on counts as another
    # after its answered its only one long cause thats what it looks like when pressing next and stuff
    def length(self):
        if self.viewable: 
            if self.state == QuestionState.INCOMPLETE: return 2
            else: return 1
        else: return 0

    # !!must be implemented!!
    # should return whatever you want
    # you will probably want to use self.state somewhere in your implementation of this
    # EXAMPLE:
    # def view(self):
    #    if self.state == QuestionState.INCORRECT: return (self, Content("WRONG"))
    #    else: return (self, self.questionDataThatIsAddedByYourSubClassThatIsAContent)
    def view(self) -> Tuple[_LearningComponent, Content]:
        return (self, Content())
    
    # !!must be implemented!!
    # you must call the superclass at the end of your implementation of it with answer being a QuestionState of what the state should change to
    # INCOMPLETE means it stays the same (probably used for invalid input)
    # CORRECT means it is correct
    # INCORRECT means it is incorrect
    # EXAMPLE:
    # def testAnswer(self, answer):
    #     if self.state != QuestionState.INCOMPLETE: return
    #     result = QuestionState.INCOMPLETE
    #     if answer == "right":
    #         result = QuestionState.CORRECT
    #     if answer == "wrong":
    #         result = QuestionState.INCORRECT
    #     super().testAnswer(result)
    def testAnswer(self, answer: Any) -> None:
        if self.state != QuestionState.INCOMPLETE: return # you will probably want to have a guard like this in a subclass 
        if answer == QuestionState.INCORRECT:
            replacement = self.replace()
            if isinstance(self.parent, QuizStructure):
                self.parent.childIsWrong(replacement) # type: ignore
        self.state = answer
        if self.parent is not None:
            self.parent.lengthDirty = True

    # !!must be implemented!!
    # create a new instance of the question but presumably different exact layout (so its fresh for the person taking it)
    # should call the superclass first then modify that 
    # EXAMPLE:
    # def replace(self):
    #     newInstance = super().replace()
    #     newInstance.yourStuff = "blablabla"
    #     return newInstance
    def replace(self) -> '_QuizQuestion':
        newInstance = self.__class__.__new__(self.__class__)
        _LearningComponent.__init__(newInstance)
        newInstance.state = QuestionState.INCOMPLETE
        return newInstance

# !!!! This is just an example implementation _QuizQuestion
# is a multiple choice question where the order of the questions are randomized with each instance of it
class RandomlyOrderedStringMultipleChoice(_QuizQuestion):
    options: List[str]
    correctIndex: int
    question: str
    showInputHint: bool
    guess: Optional[int]
    attempt: int

    # question - string
    # correct - the correct answer
    # incorrect_list - all the wrong answers
    # title - defaults to question - will be empty if "" is used
    def __init__(self, question: str, correct: str, incorrect_list: List[str], title: Optional[str] = None):
        super().__init__(title if title is not None else question)

        list = copy.copy(incorrect_list)
        random.shuffle(list)
        correctIndex = random.randint(0, len(list))
        list.insert(correctIndex, correct)

        self.options = list
        self.correctIndex = correctIndex
        self.question = question
        self.showInputHint = False # if the question should show the user a message about the valid types of input
        self.guess = None # previous guess so it can be referenced when answered
        self.attempt = 1

    # !! for internal use of this class only !!
    # returns a string that is most basic form of the question, which additional things are added to depending on state
    def viewBasic(self) -> str:
        result = self.question
        if self.attempt > 1:
            result += f" (Attempt {self.attempt})"
        for i in range(len(self.options)):
            result += f"\n{i+1}) {self.options[i]}"
        return result
    
    # returns the tuple for viewing (as defined by _LearningContent (a _LearningComponent to redirect to and the content (a string in this example) to display))
    def view(self) -> Tuple[_LearningComponent, Content]:
        # switch from match/case
        if self.state == QuestionState.INCOMPLETE:
            if self.showInputHint:
                return (self, Content(self.viewBasic() + f"\nYour answer must be an int between 1 and {len(self.options)}"))
            else:
                return (self, Content(self.viewBasic()))
        if self.state == QuestionState.CORRECT:
            return (self, Content(f"Correct!\nThe answer was {self.correctIndex + 1}\n{self.viewBasic()}"))
        if self.state == QuestionState.INCORRECT:
            return (self, Content(f"Incorrect.\nYou guessed: {self.guess}\n But the answer was {self.correctIndex + 1}\n{self.viewBasic()}"))
    
    # implementation as described by _QuizQuestion
    # take answer as number input - validates it
    # the number is used to choose which choice
    # if the choice is right its correct
    # if its wrong its incorrect
    def testAnswer(self, answer: int) -> None:
        if self.state != QuestionState.INCOMPLETE: return
        # some input validation
        if answer > len(self.options) + 1 or answer < 1:
            self.showInputHint = True
            result = QuestionState.INCOMPLETE
        else:
            self.guess = answer
            if answer == self.correctIndex + 1:
                result = QuestionState.CORRECT
            else:
                result = QuestionState.INCORRECT
        
        super().testAnswer(result)

    # create a new instance of the question but different
    # mixes the the options as a change to the question
    # returns that new question
    def replace(self) -> 'RandomlyOrderedStringMultipleChoice':
        newInstance: 'RandomlyOrderedStringMultipleChoice' = super().replace() # type: ignore
        # re-randomize the options while keeping track of the correctIndex
        list = copy.copy(self.options)
        correct = list.pop(self.correctIndex)
        random.shuffle(list)
        correctIndex = random.randint(0, len(list))
        list.insert(correctIndex, correct)
        # set those values in the copied instance
        newInstance.correctIndex = correctIndex
        newInstance.options = list
        # set the other values in the copied instance
        newInstance.showInputHint = False
        newInstance.guess= None
        newInstance.attempt = self.attempt + 1
        newInstance.question = self.question
        # return the copied instance
        return newInstance
