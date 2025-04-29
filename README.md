# pruning_interface
This files serves as documentation for the pruning interface which we create using [PySide2](https://pypi.org/project/PySide2/#:~:text=PySide2%20is%20the%20official%20Python,and%20an%20open%20design%20process.), and [OpenGL](https://pypi.org/project/PyOpenGL/). For rendering shaders, we utilize version 330 core of glsl. 

## Launching the Interface
The interface is launched using one file in the _scripts_ folder. To get to the folder, in the terminal, type in 

```cd scripts``` 

To launch the interface, in a command prompt window, type:

```python DrawTest.py```

This will launch the initial window screen after prompting the user three questions:
1. What is the Participant's ID?
2. What ordering: (A)t End or (B)uild Off?
3. What guideline: (S)patial or (R)ule?


Based on the information input by the user, the system determines which workflow to grab from _workflow.json_.

### Example Run Prompt
For example, if the user wants to run the _Spatial At End_ condition for participant number 3, then for each question, the user inputs:
1. 3
2. a
3. s

IF you want to run the "Test" workflow, when editing data, then you must input "test" for the ordering and guideline prompts. 
If no input value matches either choice or "test" for the ordering or guideline, the system will prompt again until the value is selected. 



### Text Rendering
To help render text, we utilize the [freetype-py](https://pypi.org/project/freetype-py/) library. 
To download use the command:

```pip install freetype-py```

This library allows us to use pre-existing fft files for different fonts to render in OpenGL. 
