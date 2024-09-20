#!/usr/bin/env python3

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
# from shiboken2 import VoidPtr

from scripts.MeshAndShaders import Mesh
from scripts.BranchGeometry import BranchGeometry
from scripts.GLWidget import GLWidget

from scripts.Course import Course, QuizMode, ModuleOrder
from scripts.Learning import _LearningComponent, _LearningContent, _LearningStructure

from typing import Dict, List, Literal, Optional, Tuple, Union

from scripts.DrawTest import Test

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





class StandardWindow(QMainWindow):

    def __init__(self, parent=None):
        # super(Window, self).__init__()
        QMainWindow.__init__(self, parent)
        # QtOpenGL.QMainWindow.__init__(self)
        # self.resize(1000, 1000)
        self.setGeometry(100, 100, 1000, 1000)
        self.setWindowTitle("Pruning Interface Window")

        self.central_widget = QWidget() # GLWidget()

        self.layout = QGridLayout(self.central_widget)
        # self.layout.setRowStretch(0, 40)

        # self.layout = QVBoxLayout(self.central_widget)
        self.hLayout = QHBoxLayout(self.central_widget)
        # self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)   

        # getting the main screen
        self.glWidget = GLWidget()
        # self.glWidget.setFixedSize(800, 800)
        # self.layout.addWidget(self.glWidget)
        self.layout.addWidget(self.glWidget, 0, 1, 3, 1) # r=0, c=1, rs = 3, cs = 1

        # UNDO BUTTON
        self.undoButton = QPushButton("Undo")
        self.undoButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.undoButton.setFixedSize(300, 100)
        # self.undoButton.setGeometry(200, 200, 150, 100)
        self.undoButton.clicked.connect(self.glWidget.undoDraw)
        # self.layout.addWidget(self.undoButton)
        self.hLayout.addWidget(self.undoButton)
        # self.layout.addWidget(self.undoButton, 0, 1, 1, 1)

        self.labelButton = QPushButton("Labels On") # Make a blank button
        self.labelButton.setStyleSheet("font-size: 50px;" "font:bold")
        self.labelButton.setCheckable(True)
        self.labelButton.setFixedSize(300, 100)
        self.labelButton.clicked.connect(self.labelButtonClicked)
        # self.undoButton.clicked.connect(self.glWidget.undoDraw)
        
        self.hLayout.addWidget(self.labelButton)
        # self.layout.addWidget(self.labelButton, 0, 1, 1, 1)
        self.layout.addLayout(self.hLayout, 0, 1, Qt.AlignTop | Qt.AlignLeft)

        # VERTICAL SLIDER
        self.vSlider = self.createSlider(horizontal=False)
        self.vSlider.valueChanged.connect(self.glWidget.setVerticalRotation)
        self.glWidget.verticalRotation.connect(self.vSlider.setValue)
        # self.layout.addWidget(self.vSlider)
        self.layout.addWidget(self.vSlider, 0, 0, 3, 1) 
        
        # HORIZONTAL SLIDER
        self.hSlider = self.createSlider(horizontal=True)
        self.hSlider.valueChanged.connect(self.glWidget.setTurnTableRotation)
        self.glWidget.turnTableRotation.connect(self.hSlider.setValue)
        self.layout.addWidget(self.hSlider, 3, 1, 1, 1) # 2 1 1 1

        # self.hLayout.addWidget(self.vSlider)
        # self.hLayout.addLayout(self.layout)
        self.viewGL = GLWidget(wholeView=True)
        self.viewGL.setFixedSize(800, 700)
        self.layout.addWidget(self.viewGL, 0, 2, 1, 1) # 1, 2, 1, 1
        self.hSlider.valueChanged.connect(self.viewGL.setTurnTableRotation)
        self.viewGL.turnTableRotation.connect(self.hSlider.setValue)

        self.vSlider.valueChanged.connect(self.viewGL.setVerticalRotation)
        self.viewGL.verticalRotation.connect(self.vSlider.setValue)
        
        # Create a QFrame for the directory and buttons column
        self.textFrame = QFrame(self.central_widget) # self.central_widget
        self.textFrame.setFrameShape(QFrame.Shape.Box)
        self.textFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.textFrame, 1, 2, 1, 1)  # Row 1, Column 1, Span 1 row and 1 column

        # Create a QVBoxLayout for the directory and buttons column
        self.directory_layout = QVBoxLayout(self.textFrame)

        # Create a QLabel to display the directory
        self.directory_label = QLabel("Your Task:")
        self.directory_label.setStyleSheet("font-size: 50px;" "font:bold")

        self.directory_layout.addWidget(self.directory_label)

        # Create a QLabel to display the task description
        self.task_label = QLabel("Draw on the tree section to prune back vigorous wood")
        self.task_label.setStyleSheet("font-size: 35px;")
        self.directory_layout.addWidget(self.task_label)
    
        self.progressFrame = QFrame(self.central_widget) # self.central_widget
        self.progressFrame.setFrameShape(QFrame.Shape.Box)
        self.progressFrame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.progressFrame, 2, 2, 1, 1)  # Row 1, Column 1, Span 1 row and 1 column

        self.progress_layout = QVBoxLayout(self.progressFrame)
        self.progress_label = QLabel("Your Progress:")
        self.progress_label.setStyleSheet("font-size: 50px;" "font:bold")

        self.progress_layout.addWidget(self.progress_label)


    def createSlider(self, horizontal=True):
        if horizontal:
            slider = QSlider(Qt.Horizontal)
        else:
            slider = QSlider(Qt.Vertical)
        # slider.setRange(0, 360) # 0 - 360*16
        slider.setRange(-30, 30)
        slider.setSingleStep(1) # 
        # slider.setPageStep(10)
        slider.setPageStep(5)
        slider.setTickPosition(QSlider.TicksBelow)

        return slider
        
    def labelButtonClicked(self):
        checked = True
        if self.labelButton.isChecked():
            self.labelButton.setText("Labels Off")
            checked = True 
        else:
            self.labelButton.setText("Labels On")
            checked = False
        self.glWidget.addLabels(checked) # activate the label check

################################################
# NAME: MainWindow
# DESCRIPTION: Defines the standard layout of our interface with:
#              the tree section on the left, whole tree in the top right,
#              your task and progress bar location on the right, and 
#              some sliders & buttons (not connected to anything)
#
#################################################

class MainWindow(QMainWindow):
    course: _LearningComponent # keeps track of where in the course it is and is used for viewing different pages and similar
    # images in ../icons/{str} have images scaled to 80px by 80px and is used for the task icons 
    # should not be accessed directly and only through self.getIcon(path)
    cachedImages: Dict[str, QPixmap]

    def __init__(self, parent=None, tree_fname=None):
        super().__init__()

        self.course: _LearningComponent = Course().next() # type: ignore

        # other images should be otherwise accessed with getIcon, and not directly
        # this image needs to be added before hand tho
        self.cachedImages = {
            "missing.png": QPixmap("../icons/missing.png").scaled(80, 80, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        }

        # self.main_window = main_window
        self.setWindowTitle("Pruning Interface Test")
        self.setGeometry(100, 100, 800, 600)

        # set up splitter
        # notably self.leftWidget and self.rightWidget are created
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


        self.tree_section_widget = Test() # GLDemo()
        # if tree_fname is not None:
        #     self.tree_section_widget = GLWidget(tree_fname) #GLWidget(self.central_widget)
        # else:
        #     tree_fname = '../tree_files/exemplarTree.obj'
        #     self.tree_section_widget = GLWidget(tree_fname)
        
        self.leftLayout.addWidget(self.tree_section_widget)  # Row 0, Column 0, Span 2 rows and 1 column

        # ADDING SLIDERS FOR THE GLWIDGET
        # self.horiz_slider = Slider(direction="horizontal")
        # self.horiz_slider.connectSlider(self.tree_section_widget, 
        #                                 SIGNAL("valueChanged(int)"), 
        #                                 self.tree_section_widget.setXRotation)

        
        # self.horiz_slider = self.create_slider("horizontal", # the direction of the 
        #                                        SIGNAL("valueChanged(int)"), 
        #                                        self.tree_section_widget.setXRotation)
        # self.layout.addWidget(self.horiz_slider, 2, 0, 1, 1)



        # self.vert_slider = self.create_slider(direction="vertical")
        # self.layout.addWidget(self.vert_slider, 0, 0, 2, 1)

        # if tree_fname is not None:
        #     self.whole_tree_view = GLWidget(tree_fname) #GLWidget(self.central_widget)
        # else:
        #     tree_fname = '../tree_files/exemplarTree.obj'
        #     self.whole_tree_view = GLWidget(tree_fname)

        self.whole_tree_view = Test(wholeView=True) #GLDemo()

        # self.whole_tree_view.setFixedSize(200, 150)

        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QLayoutItem.html#PySide2.QtWidgets.PySide2.QtWidgets.QLayoutItem.heightForWidth
        self.whole_tree_view.heightForWidth = lambda arg__1: (arg__1 >> 1) + (arg__1 >> 2) # it is strongly recommended to cache, but in practice it is just as laggy as this so im not gonna
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

        # progress bars are dynamicly created and edited
        # they are put into progressbar_layout layout
        # they are stored in progressbars arry
        self.progressbar_layout =  QVBoxLayout()
        self.directory_layout.addLayout(self.progressbar_layout)
        # currently displayed progress bars are stored here
        self.progressbars: List[Tuple[QLabel, QProgressBar]] = []

        # make it so anything above it is pushed up and anything below it is pushed down
        self.directory_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # create task icon and description
        self.task = QHBoxLayout()
        self.taskImage = QLabel()
        self.taskImage.setFixedWidth(80)
        self.taskImage.setFixedHeight(80)
        self.taskImagePixMap = self.getIcon("missing.png")
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
        # does not do anything right now
        self.help_button = QPushButton("Help")
        # self.help_button.clicked.connect(self.show_help)
        self.rightLayout.addWidget(self.help_button)

        self.update_from_course()
       
    # gets an icon from ../icons/{path}
    # does caching and resizing the icon to 80x80 for you
    def getIcon(self, path: str) -> QPixmap:
        if path not in self.cachedImages:
                loaded = QPixmap(f"../icons/{path}")
                if loaded.isNull():
                    self.cachedImages[path] = self.cachedImages["missing.png"]
                else:
                    self.cachedImages[path] = loaded.scaled(80, 80, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        return self.cachedImages[path]

    # triggered with the next button
    # will move to the next viewable content then display it
    def next_from_course(self):
        next = self.course.next()
        if next is None:
            print("you reached the end!")
        else:
            self.redirect(next)
    
    # triggered with the previous button
    # will move to the next viewable content then display it
    def prev_from_course(self):
        prev = self.course.prev()
        if prev is None:
            print("you reached the beginning!")
        else:
            self.redirect(prev)
    
    # updates various things
    # does the view (currently just prints (in the future should do something other than printing here) the content and follows the redirect)
    # updates the progress bars
    # updates task marker and related
    def update_from_course(self):
        #Content
        if isinstance(self.course, _LearningContent):
            if self.course.viewable:
                (redirect, content) = self.course.view()
                self.redirect(redirect, False)
                print(content.content)

        #Progressbars
        progressbars = self.course.getProgress()[::-1]

        for i in range(len(progressbars)):
            if i < len(self.progressbars):# edit
                (label, bar) = self.progressbars[i]
                label.setText(progressbars[i].title) 
                bar.setValue(int(progressbars[i].value(0.5) * 100))
                bar.considered = progressbars[i].owner
            else: #add 
                label = QLabel(progressbars[i].title, self)
                bar = ProgressBarWithContextMenuToSkip(progressbars[i].owner, self)
                bar.setValue(int(progressbars[i].value(0.5) * 100))
                self.progressbar_layout.addWidget(label)
                self.progressbar_layout.addWidget(bar)
                self.progressbars.append((label, bar))


        for _ in range(len(progressbars), len(self.progressbars)): # remove
            (label, bar) = self.progressbars.pop(len(progressbars))
            self.progressbar_layout.removeWidget(label)
            self.progressbar_layout.removeWidget(bar)
            label.setParent(None) # type: ignore
            bar.setParent(None) # type: ignore
            # label.deleteLater()
            # bar.deleteLater()

        # Marker
        marker = self.course.getProgressMarker()
        if marker is None:
            self.taskDescription.setText("")
            self.taskImage.setPixmap(self.cachedImages["missing.png"])
        else:
            self.taskDescription.setText(marker.description)
            self.taskImage.setPixmap(self.getIcon(marker.image))

    # will follow a redirect to a different content and then update the display
    # should be used instead of self.course = foobar unless you have a good reason
    # update is optional and if its false will never update, if its true will always update, and if its None (default) will update if its needed
    def redirect(self, place: _LearningComponent, update: Optional[bool] = None):
        if place != self.course:
            self.course = place
            if update is not False: self.update_from_course()
        else:
            if update is True: self.update_from_course()
    
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
    
    # adds a context menu with options to change the course/presentation mode
    def contextMenuEvent(self, event):
        contextMenu = QMenu(self)

        actualCourse:Course = self.course.getRoot() # type: ignore
        
        # Add custom actions
        at_the_end = QAction('At The End', self)
        build_off = QAction('Build Off', self)
        spatial = QAction('Spatial', self)
        rule = QAction('Rule', self)

        # makes it effectively 2 radio buttons even tho they are technically checkboxes
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
        
        contextMenu.addAction(at_the_end)
        contextMenu.addAction(build_off)
        contextMenu.addSeparator()
        contextMenu.addAction(spatial)
        contextMenu.addAction(rule)
        
        # Show the context menu at the position of the right-click
        contextMenu.exec_(event.globalPos())

    # if it is no longer connected to the root of the course (a component that is being viewed got removed)
    # then redirect to the beginning of the course to prevent unexpected behavior
    # update is similar to redirect's update
    def ensureConnectivityToCourse(self, update: Optional[bool] = None):
        if not self.course.isConnectedToRoot():
            print("redirecting to the beginning because connectivity to root was lost")
            self.redirect(self.course.getRoot().firstChild().next(), update) # type: ignore
        if update is True:
            self.update_from_course()

    # all four are the same:
    # change the mode
    # ensure not disconnected
    # update display
    def switchToAtTheEnd(self):
        self.course.getRoot().quizMode = QuizMode.AT_THE_END # type: ignore
        self.ensureConnectivityToCourse(True)

    def switchToBuildOff(self):
        self.course.getRoot().quizMode = QuizMode.BUILD_OFF # type: ignore
        self.ensureConnectivityToCourse(True)

    def switchToSpatial(self):
        self.course.getRoot().moduleOrder = ModuleOrder.SPATIAL # type: ignore
        self.ensureConnectivityToCourse(True)

    def switchToRule(self):
        self.course.getRoot().moduleOrder = ModuleOrder.RULE # type: ignore
        self.ensureConnectivityToCourse(True)

# should be only used in the main directory
# they are normal progress bars except,
# they can be right clicked to skip it or to go to the beginning of it
class ProgressBarWithContextMenuToSkip(QProgressBar):
    main: MainWindow # so that it can call .redirect(foobar)
    considered: _LearningStructure # so it knows what it is so that it knows where to redirect to 

    def __init__(self, considered: _LearningStructure, main: MainWindow):
        super().__init__(main)
        self.main = main
        self.considered = considered
    
    def contextMenuEvent(self, event):
        context_menu = QMenu(self)

        beginning_action = QAction("Beginning", self)
        skip_action = QAction("Skip", self)

        def goToBeginning():
            beginning = self.considered.firstChild()
            if beginning is None: return
            # doesn't go to first child but first viewable thing starting from the first child - this if does the later part
            if not beginning.viewable:
                beginning = beginning.next()
            
            self.main.redirect(beginning)
        
        def skip():
            parent = self.considered.parent
            # you cant skip the top level progress bar (because there is nothing after it)
            if parent is None: return
            next = parent.intoNext(self.considered)
            # if it is the last, then you cant skip it
            # but instead of skipping it will go to the end of it instead
            end = self.considered.lastChild()
            if end is not None: end = end.intoPrev(parent)

            if next is None and end is None: return
            self.main.redirect(next or end)

        beginning_action.triggered.connect(goToBeginning)
        skip_action.triggered.connect(skip)

        context_menu.addAction(beginning_action)
        # you cant skip the top level progress bar (because there is nothing after it)
        if (self.considered.parent is not None): context_menu.addAction(skip_action)

        # Show the context menu at the position of the right-click
        context_menu.exec_(event.globalPos())



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
