# version 330 core

in vec3 TexCoords;
out vec4 FragColor;

uniform samplerCube gCubeMapTexture;


void main()
{
    FragColor = texture(gCubeMapTexture, TexCoords);
    //FragColor = vec4(abs(fract(TexCoords)), 1);
}