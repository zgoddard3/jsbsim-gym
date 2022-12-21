#version 330 core

in vec3 Normal;

uniform vec3 color;
uniform vec3 lightDir;

out vec4 FragColor;

void main() {
    float ambient = .2f;
    FragColor = vec4(color * (ambient + max(dot(normalize(lightDir), -Normal), 0.0f)), 1.0f);
}