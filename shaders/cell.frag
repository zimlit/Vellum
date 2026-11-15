// Text pass: coverage from the atlas times the glyph color, premultiplied,
// blended over what's there. Quads are pixel-aligned, so each fragment maps
// to exactly one atlas texel; no sampler or filtering needed.
#version 460
#extension GL_GOOGLE_include_directive : require
#include "cell.glsl"

layout(set = 0, binding = 0, rgba8) uniform readonly image2D atlas;

layout(location = 0) in vec2 local;
layout(location = 1) flat in uvec2 atlas_origin;
layout(location = 2) flat in vec4 color;
layout(location = 3) flat in uint flags;

layout(location = 0) out vec4 frag;

void main() {
    float coverage = 1.0;
    if ((flags & FLAG_SOLID) == 0u) {
        // Linear coverage, blended as is (gamma-space blending, like most
        // terminals). Gamma/contrast tuning goes here later.
        coverage = imageLoad(atlas, ivec2(atlas_origin) + ivec2(floor(local))).r;
    }
    float alpha = color.a * coverage;
    frag = vec4(color.rgb * alpha, alpha);
}
