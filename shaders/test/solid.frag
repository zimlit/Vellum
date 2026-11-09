// Test: fill with the color in push constants.
#version 460

layout(push_constant) uniform Push {
    vec4 color;
};

layout(location = 0) out vec4 frag;

void main() {
    frag = color;
}
