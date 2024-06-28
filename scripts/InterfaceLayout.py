#######################################
# OWNER: OSURobotics
# PURPOSE: Defining the layout of our pruning interface
######################################

import sys
sys.path.append('../')


from PySide2 import QtCore, QtGui
from PySide2.QtWidgets import QApplication, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QMainWindow, QFrame, QGridLayout, QPushButton, QOpenGLWidget, QProgressBar, QSpacerItem, QSizePolicy, QSplitter
from PySide2.QtCore import Qt, Signal, SIGNAL, SLOT, QPoint
from PySide2.QtOpenGL import QGLWidget
from PySide2.QtGui import QPixmap, QOpenGLVertexArrayObject, QOpenGLBuffer, QOpenGLShaderProgram, QOpenGLShader, QOpenGLContext, QVector4D, QMatrix4x4
from shiboken2 import VoidPtr
from scripts.MeshAndShaders import Mesh
from scripts.BranchGeometry import BranchGeometry
from scripts.GLWidget import GLWidget
from scripts.Course import Course

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

        # if the interface gets laggy (even when the whole_tree_view does not have a changing width), then try just disabling this
        # https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QLayoutItem.html#PySide2.QtWidgets.PySide2.QtWidgets.QLayoutItem.heightForWidth
        self.whole_tree_view.cached_width = 0
        self.whole_tree_view.cached_height = 0
        def whole_tree_view_height_for_width(w):
            if self.whole_tree_view.cached_width != w:
                h = w * 0.75
                self.whole_tree_view.cached_width = w
                self.whole_tree_view.cached_height = h
            return self.whole_tree_view.cached_height

        self.whole_tree_view.heightForWidth = whole_tree_view_height_for_width
        self.whole_tree_view.hasHeightForWidth = lambda: True
        # self.whole_tree_view.width 
        # self.whole_tree_view.setFixedWidth(200)

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

        temp_progressbar = QVBoxLayout()
        temp_progressbar_label = QLabel("Course:")
        temp_progressbar_bar = QProgressBar()
        temp_progressbar.addWidget(temp_progressbar_label)
        temp_progressbar.addWidget(temp_progressbar_bar)
        self.progressbars = [(temp_progressbar, temp_progressbar_label, temp_progressbar_bar)]

        self.progressbar_layout.addLayout(temp_progressbar)

        self.directory_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.task = QHBoxLayout()
        self.taskImage = QLabel()
        self.taskImage.setFixedWidth(80)
        self.taskImage.setFixedHeight(80)
        self.taskImagePixMap = QPixmap("../icons/missing.png").scaled(80, 80, Qt.IgnoreAspectRatio, Qt.FastTransformation)
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
        # self.previous_button.clicked.connect(self.show_previous_image)
        self.prev_next_layout.addWidget(self.previous_button)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.update_from_course)
        self.prev_next_layout.addWidget(self.next_button)

        # Create a Help button
        self.help_button = QPushButton("Help")
        # self.help_button.clicked.connect(self.show_help)
        self.rightLayout.addWidget(self.help_button)
       
    def update_from_course(self):
        print("e")

    
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
    # NAME: createScaleSlider
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
