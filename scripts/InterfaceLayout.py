#!/usr/bin/env python3

#######################################
# OWNER: OSURobotics
# PURPOSE: Defining the layout of our pruning interface
######################################

import sys
sys.path.append('../')


from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4
from shiboken2 import VoidPtr
from scripts.MeshAndShaders import Mesh
from scripts.BranchGeometry import BranchGeometry
from scripts.GLWidget import GLWidget
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
#              your task and progress bar locfation on the right, and 
#              some sliders & buttons (not connected to anything)
#
#################################################

class MainWindow(QMainWindow):
    def __init__(self, parent=None, tree_fname=None):
        super().__init__()

        # self.main_window = main_window
        self.setWindowTitle("Pruning Interface Test")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget() # GLWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout(self.central_widget)

        self.tree_section_widget = Test() # GLDemo()
        # if tree_fname is not None:
        #     self.tree_section_widget = GLWidget(tree_fname) #GLWidget(self.central_widget)
        # else:
        #     tree_fname = '../tree_files/exemplarTree.obj'
        #     self.tree_section_widget = GLWidget(tree_fname)
        
        self.layout.addWidget(self.tree_section_widget, 0, 0, 2, 1)  # Row 0, Column 0, Span 2 rows and 1 column

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

        self.whole_tree_view.setFixedSize(200, 150)
        self.layout.addWidget(self.whole_tree_view, 0, 1, 1, 1)  # Row 0, Column 1, Span 1 row and 1 column

        # Create a QFrame for the directory and buttons column
        self.frame = QFrame(self.central_widget) # self.central_widget
        self.frame.setFrameShape(QFrame.Shape.Box)
        self.frame.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.frame, 1, 1, 1, 1)  # Row 1, Column 1, Span 1 row and 1 column

        # Create a QVBoxLayout for the directory and buttons column
        self.directory_layout = QVBoxLayout(self.frame)

        # Create a QLabel to display the directory
        self.directory_label = QLabel("Your Task:")
        self.directory_layout.addWidget(self.directory_label)

        # Create a QLabel to display the task description
        self.task_label = QLabel("Task description goes here")
        self.directory_layout.addWidget(self.task_label)


        # Create buttons for navigation
        self.previous_button = QPushButton("Previous")
        # self.previous_button.clicked.connect(self.show_previous_image)
        self.directory_layout.addWidget(self.previous_button)

        self.next_button = QPushButton("Next")
        # self.next_button.clicked.connect(self.show_next_image)
        self.directory_layout.addWidget(self.next_button)

        # Create a Help button
        self.help_button = QPushButton("Help")
        # self.help_button.clicked.connect(self.show_help)
        self.directory_layout.addWidget(self.help_button)
       
        
    
    def create_slider(self, direction, changedSignal, setSlot):
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
       

#######################################################
# CLASS NAME: Slider
# DESCRIPTION: Creates a slider on the widegt screen
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
    def createLengthSlider(self):
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
    
    def createRotationSlider(self):
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
    # NAME: createscaleSlider
    # DESCRIPTION: creates a slider for controlling the scale/size of the branch
    # INPUT: None
    # OUTPUT: 
    #   - slider: a slider that is either vertical or horizontal for controlling the scale/size of the branch
    ###########################################
    def createScaleSlider(self):
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
    def setValue(self, val):
        self.slider.setValue(val)

    def getSlider(self):
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

    
    def createRotationSlider(self, direction):
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

    def createScaleSlider(self, direction):
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
