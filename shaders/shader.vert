#version 330 core

// set the attributes we are receiving and where they are stored 
layout (location = 0) in vec2 aTexCoord;
layout (location = 1) in vec3 aNormal; // location = 0
layout (location = 2) in vec3 vertexPos; // location = 1


// get the uniform values for defining our viewing frustrum
uniform mat4 view;
uniform mat4 model;
uniform mat4 projection; 

// out vec4 treeColor; // passed to the fragment shader
out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoord;

void main()
{
    
    FragPos = vec3(model * vec4(vertexPos, 1.0)); // translates to world space
    Normal = aNormal; 
    gl_Position = projection * view * model * vec4(vertexPos, 1.0); 
    TexCoord = aTexCoord; // set the coordinates
} 