#######################################
# OWNER: OSURobotics
# PURPOSE: Defining the layout of our pruning interface
######################################

import sys
sys.path.append('../')


from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget, QProgressBar, QSpacerItem, QSizePolicy, QSplitter, QAction, QMenu
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QPixmap, QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4
from shiboken2 import VoidPtr

from scripts.MeshAndShaders import Mesh
from scripts.BranchGeometry import BranchGeometry
from scripts.GLWidget import GLWidget

from scripts.Course import Course, QuizMode, ModuleOrder
from scripts.Learning import _LearningComponent, _LearningContent

from typing import List, Literal, Tuple, Union

# from PySide2.shiboken2 import VoidPtr

# from PyQt6 import QtCore      # core Qt functionality
# from PyQt6 import QtGui       # extends QtCore with GUI functionality
# from PyQt6 import QtOpenGL    # provides QGLWidget, a special OpenGL QWidget

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality
# from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.arrays import vbo
import pywavefront
import numpy as np
import ctypes                 # to communicate with c code under the hood



""" 
TODO:
DONE Progress Marker
     Right click progress bars to skip
        bar.considered is incorrect right now
        either doesn't update properly or never got set right
DONE Right click Course to change mode
     Review Comments on Learning.py
DONE Fix not getting to 100% in at the end mode
     Check other todo list
     Add comments to Course.py
     clean up this file some more
     PUSH!
"""



################################################
# NAME: MainWindow
# DESCRIPTION: Defines the standard layout of our interface with:
#              the tree section on the left, whole tree in the top right,
#              your task and progress bar location on the right, and 
#              some sliders & buttons (not connected to anything)
#
#################################################

class MainWindow(QMainWindow):
    def __init__(self, parent=None, tree_fname=None):
        super().__init__()

        self.course: _LearningComponent = Course().next()

        # self.main_window = main_window
        self.setWindowTitle("Pruning Interface Test")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget() # GLWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout = QHBoxLayout(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.central_layout.addWidget(self.splitter)
        self.leftWidget = QWidget()
        self.leftLayout = QVBoxLayout(self.leftWidget)
        self.splitter.addWidget(self.leftWidget)
        self.rightWidget = QWidget()
        self.rightLayout = QVBoxLayout(self.rightWidget)
        self.splitter.addWidget(self.rightWidget)
        self.splitter.setSizes([self.width() - 200, 200])


        if tree_fname is not None:
            self.tree_section_widget = GLWidget(tree_fname) #GLWidget(self.central_widget)
        else:
            tree_fname = '../tree_files/exemplarTree.obj'
            self.tree_section_widget = GLWidget(tree_fname)
        
        self.leftLayout.addWidget(self.tree_section_widget)  # Row 0, Column 0, Span 2 rows and 1 column

        # ADDING SLIDERS FOR THE GLWIDGET
        # self.horiz_slider = Slider(direction="horizontal")
        # self.horiz_slider.connectSlider(self.tree_section_widget, 
        #                                 SIGNAL("valueChanged(int)"), 
        #                                 self.tree_section_widget.setXRotation)

        self.horiz_slider = self.create_slider("horizontal", # the direction of the 
                                               SIGNAL("valueChanged(int)"), 
                                               self.tree_section_widget.setXRotation)
        self.leftLayout.addWidget(self.horiz_slider)


        # self.vert_slider = self.create_slider(direction="vertical")
        # self.layout.addWidget(self.vert_slider, 0, 0, 2, 1)

        if tree_fname is not None:
            self.whole_tree_view = GLWidget(tree_fname) #GLWidget(self.central_widget)
        else:
            tree_fname = '../tree_files/exemplarTree.obj'
            self.whole_tree_view = GLWidget(tree_fname)

        # self.whole_tree_view.setFixedSize(200, 150)

        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QLayoutItem.html#PySide2.QtWidgets.PySide2.QtWidgets.QLayoutItem.heightForWidth
        self.whole_tree_view.heightForWidth = lambda w: (w >> 1) + (w >> 2) # it is strongly recommended to cache, but in practice it is just as laggy as this so im not gonna
        # if the interface gets laggy (even when the whole_tree_view does not have a changing width), then try removing this line of code ↓
        self.whole_tree_view.hasHeightForWidth = lambda: True

        # self.whole_tree_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.rightLayout.addWidget(self.whole_tree_view)

        # Create a QFrame for the directory and buttons column
        self.frame = QFrame() # self.central_widget
        self.frame.setFrameShape(QFrame.Shape.Box)
        self.frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.rightLayout.addWidget(self.frame)

        # Create a QVBoxLayout for the directory and buttons column
        self.directory_layout = QVBoxLayout(self.frame)

        self.progressbar_layout =  QVBoxLayout()
        self.directory_layout.addLayout(self.progressbar_layout)

        self.progressbars: List[Tuple[QLabel, QProgressBar]] = []

        self.directory_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.cachedImages = {
            "missing.png": QPixmap("../icons/missing.png").scaled(80, 80, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        }
        self.task = QHBoxLayout()
        self.taskImage = QLabel()
        self.taskImage.setFixedWidth(80)
        self.taskImage.setFixedHeight(80)
        self.taskImagePixMap = self.cachedImages["missing.png"]
        self.taskImage.setPixmap(self.taskImagePixMap)
        self.task.addWidget(self.taskImage)
        self.taskDescription = QLabel("This is a description that should be written but isn't. Or, maybe it is. Who knows? I don't. Should I? Yikes, I better get on figuring that one out.")
        self.taskDescription.setWordWrap(True)
        self.task.addWidget(self.taskDescription)
        self.directory_layout.addLayout(self.task)

        self.prev_next_layout = QHBoxLayout()
        self.directory_layout.addLayout(self.prev_next_layout)

        # Create buttons for navigation
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.prev_from_course)
        self.prev_next_layout.addWidget(self.previous_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_from_course)
        self.prev_next_layout.addWidget(self.next_button)

        # Create a Help button
        self.help_button = QPushButton("Help")
        # self.help_button.clicked.connect(self.show_help)
        self.rightLayout.addWidget(self.help_button)

        self.update_from_course()
       
    def next_from_course(self):
        next = self.course.next()
        if next is None:
            print("you reached the end!")
        else:
            self.course = next
            self.update_from_course()

    
    def prev_from_course(self):
        prev = self.course.prev()
        if prev is None:
            print("you reached the beginning!")
        else:
            self.course = prev
            self.update_from_course()
    
    def update_from_course(self):
        #Content
        if isinstance(self.course, _LearningContent):
            if self.course.viewable:
                (redirect, content) = self.course.view()
                self.course = redirect
                print(content)

        #Progressbars
        def progressBarContextMenu(self, event, main):
            context_menu = QMenu(self)

            beginning_action = QAction("Beginning", self)
            skip_action = QAction("Skip", self)

            def goToBeginning():
                beginning = self.considered.firstChild()
                main.redirect(beginning)
            
            def skip():
                end = self.considered.lastChild()
                beyond = end.next()
                main.redirect(beyond or end)

            beginning_action.triggered.connect(goToBeginning)
            skip_action.triggered.connect(skip)

            context_menu.addAction(beginning_action)
            context_menu.addAction(skip_action)

            context_menu.exec_(event.globalPos())

        progressbars = self.course.getProgress()[::-1]

        for i in range(len(progressbars)):
            if i < len(self.progressbars):# edit
                (label, bar) = self.progressbars[i]
                label.setText(progressbars[i].title) 
                bar.setValue(progressbars[i].value(0.5) * 100)
                bar.considered = progressbars[i].owner
            else: #add 
                label = QLabel(progressbars[i].title, self)
                bar = QProgressBar(self)
                bar.setValue(progressbars[i].value(0.5) * 100)
                self.progressbar_layout.addWidget(label)
                self.progressbar_layout.addWidget(bar)
                self.progressbars.append((label, bar))
                bar.considered = progressbars[i].owner
                bar.contextMenuEvent = lambda event: progressBarContextMenu(bar, event, self)


        for _ in range(len(progressbars), len(self.progressbars)): # remove
            (label, bar) = self.progressbars.pop(len(progressbars))
            self.progressbar_layout.removeWidget(label)
            self.progressbar_layout.removeWidget(bar)
            label.setParent(None)
            bar.setParent(None)
            # label.deleteLater()
            # bar.deleteLater()

        # Marker
        marker = self.course.getProgressMarker()
        if marker is None:
            self.taskDescription.setText("")
            self.taskImage.setPixmap(self.cachedImages["missing.png"])
        else:
            self.taskDescription.setText(marker.description)
            if marker.image not in self.cachedImages:
                loaded = QPixmap(f"../icons/{marker.image}")
                if loaded.isNull():
                    self.cachedImages[marker.image] = self.cachedImages["missing.png"]
                else:
                    self.cachedImages[marker.image] = loaded.scaled(80, 80, Qt.IgnoreAspectRatio, Qt.FastTransformation)
            self.taskImage.setPixmap(self.cachedImages[marker.image])

    def redirect(self, place: _LearningComponent):
        self.course = place
        self.update_from_course()

    
    def create_slider(self, direction: Literal["horizontal", "vertical"], changedSignal, setSlot):
        slider = QSlider()
        if direction == "horizontal":
            slider.setOrientation(Qt.Horizontal)
        else:
            slider.setOrientation(Qt.Vertical)
        
        # slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setMinimum(0)
        slider.setMaximum(360 * 16)    

        self.tree_section_widget.connect(slider, SIGNAL("valueChanged(int)"), setSlot)
        # QWidget.connect(self.tree_section_widget, changedSignal, slider, SLOT("setValue(int)")) 
    
        return slider
    
    def contextMenuEvent(self, event):
        # Create a custom context menu
        contextMenu = QMenu(self)

        actualCourse:Course = self.course.getRoot()
        
        # Add custom actions
        at_the_end = QAction('At The End', self)
        build_off = QAction('Build Off', self)
        spatial = QAction('Spatial', self)
        rule = QAction('Rule', self)

        at_the_end.setCheckable(True)
        build_off.setCheckable(True)
        spatial.setCheckable(True)
        rule.setCheckable(True)

        match actualCourse.quizMode:
            case QuizMode.AT_THE_END:
                at_the_end.setChecked(True)
                at_the_end.setDisabled(True)
            case QuizMode.BUILD_OFF:
                build_off.setChecked(True)
                build_off.setDisabled(True)

        match actualCourse.moduleOrder:
            case ModuleOrder.SPATIAL:
                spatial.setChecked(True)
                spatial.setDisabled(True)
            case ModuleOrder.RULE:
                rule.setChecked(True)
                rule.setDisabled(True)

        at_the_end.triggered.connect(self.switchToAtTheEnd)
        build_off.triggered.connect(self.switchToBuildOff)
        spatial.triggered.connect(self.switchToSpatial)
        rule.triggered.connect(self.switchToRule)

        # .setCheckable(True)
        
        contextMenu.addAction(at_the_end)
        contextMenu.addAction(build_off)
        contextMenu.addSeparator()
        contextMenu.addAction(spatial)
        contextMenu.addAction(rule)
        
        # Show the context menu at the position of the right-click
        contextMenu.exec_(event.globalPos())

    def switchToAtTheEnd(self):
        self.course.getRoot().quizMode = QuizMode.AT_THE_END
        self.update_from_course()

    def switchToBuildOff(self):
        self.course.getRoot().quizMode = QuizMode.BUILD_OFF
        self.update_from_course()

    def switchToSpatial(self):
        self.course.getRoot().moduleOrder = ModuleOrder.SPATIAL
        self.update_from_course()

    def switchToRule(self):
        self.course.getRoot().moduleOrder = ModuleOrder.RULE
        self.update_from_course()


#######################################################
# CLASS NAME: Slider
# DESCRIPTION: Creates a slider on the widget screen
#######################################################
class Slider(QWidget):
    def __init__(self, connectWidget, horiz=True, rDir=None, sDir=None, label="Slider", sliderType=1):
        self.horiz = horiz
        self.rDir = rDir        # rotation
        self.sDir = sDir        # scale
        self.sliderType = sliderType    # 1 = rotation, 2 = scale
        self.connectWidget = connectWidget
        # self.changedSignal = changedSignal
        # self.setterSlot = setterSlot

        self.label = QLabel(label)
        self.label.setFont(QtGui.QFont("Sanserif", 8))

        if sliderType == 1:
            self.slider = self.createRotationSlider()
        elif sliderType == 2:
            self.slider = self.createScaleSlider() 
        else:
            self.slider = self.createLengthSlider()

        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label)
        layout.addWidget(self.slider)

    ##########################################
    #
    # NAME: createLengthSlider
    # DESCRIPTION: creates a slider for controlling the length of the branch
    # INPUT: None
    # OUTPUT: 
    #   - slider: a slider that is either vertical or horizontal for controlling the branch length
    ###########################################
    def createLengthSlider(self) -> QSlider:
        slider = QSlider()
        if self.horiz:
            slider.setOrientation(Qt.Horizontal)
        else:
            slider.setOrientation(Qt.Vertical)
        
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(10)            # 16
        slider.setPageStep(100)         # 15 * 16
        slider.setTickInterval(100)     # 15 * 16
        slider.setMinimum(1)           # should always have some length      
        slider.setMaximum(1000)         # 360 * 16

        slider.valueChanged.connect(self.connectWidget.setLength)
        self.connectWidget.mesh.lengthChanged.connect(slider.setValue)        
        return slider

    ##########################################
    #
    # NAME: createRotationSlider
    # DESCRIPTION: creates a slider for controlling the rotation of the branch
    # INPUT: None
    # OUTPUT: 
    #   - slider: a slider that is either vertical or horizontal for controlling the rotation of the branch
    ###########################################
    
    def createRotationSlider(self) -> QSlider:
        slider = QSlider()
        if self.horiz:
            slider.setOrientation(Qt.Horizontal)
        else:
            slider.setOrientation(Qt.Vertical)
        
        # slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setMinimum(0)
        slider.setMaximum(360 * 16) 


        if self.rDir is None:
            print("Rotation direction not assigned. Slider not connected")
        elif self.rDir == 1:
            slider.valueChanged.connect(self.connectWidget.setXRotation)
            self.connectWidget.mesh.xRotationChanged.connect(slider.setValue)
        elif self.rDir == 2:
            slider.valueChanged.connect(self.connectWidget.setYRotation)
            self.connectWidget.mesh.yRotationChanged.connect(slider.setValue)
        else:
            slider.valueChanged.connect(self.connectWidget.setZRotation)
            self.connectWidget.mesh.zRotationChanged.connect(slider.setValue)
    

        # self.connectWidget.connect(self.slider, self.changedSignal, self.setterSlot)
        # self.connect(self.connectWidget.mesh, self.changedSignal, self.slider, SLOT("setValue(int)"))   
 
        return slider
    


    ##########################################
    #
    # NAME: createScaleSlider
    # DESCRIPTION: creates a slider for controlling the scale/size of the branch
    # INPUT: None
    # OUTPUT: 
    #   - slider: a slider that is either vertical or horizontal for controlling the scale/size of the branch
    ###########################################
    def createScaleSlider(self) -> QSlider:
        slider = QSlider()
        if self.horiz:
            slider.setOrientation(Qt.Horizontal)
        else:
            slider.setOrientation(Qt.Vertical)
        
        # slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(1) # 16
        slider.setPageStep(10) # 15 * 16
        slider.setTickInterval(10) # 15 * 16
        slider.setMinimum(1) # 0
        slider.setMaximum(100) # 360 * 16

        if self.sDir is None:
            print("Scale direction not assigned. Slider not connected")
        elif self.sDir == 1: # scale
            # print("Connecting slider to branch length scaling")
            slider.valueChanged.connect(self.connectWidget.setXScale)
            self.connectWidget.mesh.xScaleChanged.connect(slider.setValue)
        else: # diameter scale
            # print("Connecting slider to branch diameter scaling")
            slider.valueChanged.connect(self.connectWidget.setDiameterScale)
            self.connectWidget.mesh.diameterChanged.connect(slider.setValue)

        return slider


    ##########################################
    #
    # NAME: setValue and getValue
    # DESCRIPTION: getters and setters for defining what value the slider is at currently and where to set the
    #              starting value
    # INPUT: 
    #   - setValue: 
    #       - val -- an int or float representing the value to start
    #   - getValue: None
    # OUTPUT: 
    #   - getSlider: a slider that is either vertical or horizontal for controlling the branch length
    #   - setSlider: None
    ###########################################
    def setValue(self, val: Union[int, float]):
        self.slider.setValue(val)

    def getSlider(self) -> QSlider:
        return self.slider
    # different functions for changing values
    




#################################
# NAME: TestWindow
# DESCRIPTION: A class for creating a test layout for including sliders that change the branch's features
#              such as length, size, and rotation
#################################

class TestWindow(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        tree_fname = '../tree_files/exemplarTree.obj'
        self.glWidget = GLWidget(tree_fname)
        # self.rotateLabel = QLabel("Turntable")
        self.rotationSlider = Slider(connectWidget=self.glWidget, 
                                     horiz=True, 
                                     rDir=2, # 1 = x, 2 = y, 3 = z
                                     sDir=None,
                                     label="Turntable Motion", 
                                     sliderType=1) # 1 = rotation, 2=scale
        # self.rotationSlider = self.createRotationSlider("horizontal")
        # self.rotationSlider.valueChanged.connect(self.glWidget.setYRotation)
        # self.glWidget.mesh.yRotationChanged.connect(self.rotationSlider.setValue)


        self.scaleSlider = Slider(connectWidget=self.glWidget,
                                  horiz=True,
                                  rDir=None,
                                  sDir=1, # 1 = sx
                                  label="Branch Scale",
                                  sliderType=2)
        self.scaleSlider.setValue(10)


        self.lengthSlider = Slider(connectWidget=self.glWidget,
                                   horiz=True,
                                   rDir=None,
                                   sDir=None,
                                   label="Branch Length",
                                   sliderType=3)
        self.lengthSlider.setValue(50)



        self.diameterSlider = Slider(connectWidget=self.glWidget,
                                     horiz=True,
                                     rDir=None,
                                     sDir=2, # 2 = branch diameter
                                     label="Branch Diameter",
                                     sliderType=2)
        self.diameterSlider.setValue(10)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.glWidget)
        mainLayout.addWidget(self.rotationSlider)
        mainLayout.addWidget(self.lengthSlider)
        # mainLayout.addWidget(self.scaleSlider)
        # mainLayout.addWidget(self.diameterSlider)

        self.setLayout(mainLayout)

    
    def createRotationSlider(self, direction: Literal["horizontal", "vertical"]) -> QSlider:
        slider = QSlider()
        if direction == "horizontal":
            slider.setOrientation(Qt.Horizontal)
        else:
            slider.setOrientation(Qt.Vertical)
        
        # slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(16)  # 16
        slider.setPageStep(15 * 16) # 15 * 16
        slider.setTickInterval(15 * 16) # 15 * 16
        slider.setMinimum(0) # 0
        slider.setMaximum(360 * 16) # 360 * 16   
    
        return slider

    def createScaleSlider(self, direction: Literal["horizontal", "vertical"]) -> QSlider:
        slider = QSlider()
        if direction == "horizontal":
            slider.setOrientation(Qt.Horizontal)
        else:
            slider.setOrientation(Qt.Vertical)
        
        # slider.setValue(0)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(10)  # 16
        slider.setPageStep(10 * 10) # 15 * 16
        slider.setTickInterval(10 * 10) # 15 * 16
        slider.setMinimum(1) # 0
        slider.setMaximum(100 * 10) # 360 * 16   
        return slider
    


    
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super(TestWindow, self).keyPressEvent(event)
