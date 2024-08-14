# version 330 core
// inspired by https://www.youtube.com/watch?v=8sVvxeKI9Pk 
// and https://learnopengl.com/Advanced-OpenGL/Cubemaps

layout (location = 2) in vec3 aPos;

out vec3 TexCoords;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    vec4 pos = projection * view * model * vec4(aPos, 1.0); 
    gl_Position = vec4(pos.x, pos.y, pos.w, pos.w); // always want z to be 1 to be behind everything, so set to w
    TexCoords = aPos;
    //TexCoords = vec3(aPos.x, aPos.y, -aPos.z); // uses a left hand coordinate system so -z
}