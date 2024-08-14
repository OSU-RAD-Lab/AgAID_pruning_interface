# version 330 core
in vec4 color;
in vec2 TexCoord;

uniform sampler2D ourTexture;

out vec4 treeColor;

void main()
{   
    //treeColor = color * texture(ourTexture, TexCoord);
    treeColor = color;
}        