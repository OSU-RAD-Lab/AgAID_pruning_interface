# version 140 
attribute highp vec3 vertexPos;
uniform mat4 mvMatrix;
uniform mat4 projMatrix;
void main(void)
{
    gl_Position = projMatrix * mvMatrix * vec4(vertexPos, 1.0);
}