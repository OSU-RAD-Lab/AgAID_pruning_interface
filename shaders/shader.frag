# version 330 core
// set the attributes we are receiving and where they are stored 

// Values inherited/pulled from the vertex shader 
in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoord; 

// lighting and texture information
uniform vec3 lightColor;
uniform vec3 lightPos;
uniform vec3 color;
uniform sampler2D ourTexture;

out vec4 treeColor; // output of the fragment shader is the color

void main()
{
    float ambientStrength = 1.0;
    vec3 ambient = ambientStrength * lightColor;
    
    //vec3 result = ambient * vec3(0.5, 0.25, 0.0);
    //treeColor = vec4(result, 1.0);
        
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Tree colors
    // 0.251, 0.212, 0.169 --> Dark Brown
    // 0.447, 0.360, 0.259 --> Light Brown (almost orange)
    // 0.239, 0.231, 0.192 
    // 0.670, 0.549, 0.416
    // 0.278, 0.219, 0.227

    vec3 result = (ambient + diffuse) * color; 
    //vec3 result = (ambient + diffuse) * vec3(0.447, 0.360, 0.259); 
    //vec3 result = ambient + diffuse;
    treeColor = vec4(result, 1.0) * texture(ourTexture, TexCoord);
    //treeColor = vec4(result, 0.2);             
}