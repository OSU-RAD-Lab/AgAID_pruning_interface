# version 330 core
layout (location = 0) in vec4 vertex;
out vec2 TexCoords;

uniform mat4 projection;
//uniform mat4 view;
//uniform mat4 model;

void main()
{
    //vec4 pos = projection * view * model * vec4(vertex.xy, 0.0, 1.0);
    vec4 pos = projection * vec4(vertex.xy, 0.0, 1.0);
    gl_Position = vec4(pos.xy, 0.0, 1.0); // render text at the nearest point on the screen (i.e., d=0.1)
    TexCoords = vertex.zw;
}