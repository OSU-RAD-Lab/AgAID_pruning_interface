# pruning_interface
This files serves as documentation for the pruning interface which we create using [PySide2](https://pypi.org/project/PySide2/#:~:text=PySide2%20is%20the%20official%20Python,and%20an%20open%20design%20process.), and [OpenGL](https://pypi.org/project/PyOpenGL/). For rendering shaders, we utilize version 330 core of glsl. 

## Launching the Interface
To launch the interface, in a command prompt window, type:

```python InterfaceMain.py```


## Text Rendering
To help render text, we utilize the [freetype-py](https://pypi.org/project/freetype-py/) library. 
To download use the command:

```pip install freetype-py```

This library allows us to use pre-existing fft files for different fonts to render in OpenGL. 