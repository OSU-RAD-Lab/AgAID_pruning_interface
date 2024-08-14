# version 330 core

layout (location = 0) in vec2 aTexCoord;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 vertexPos;

out vec4 color;
out vec2 TexCoord;

void main()
{
    TexCoord = aTexCoord;
    Normal = aNormal;
    color = vec4(0.5, 0.25, 0.0, 1.0); 
    gl_Position = vec4(vertexPos, 1.0);
}  