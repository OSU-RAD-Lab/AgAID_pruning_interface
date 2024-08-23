# version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec4 color;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    color = vec4(1.0, 0.0, 0.0, 1.0);
} 