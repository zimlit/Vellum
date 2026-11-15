// Text pass: one instance per glyph, six vertices (two triangles) each. The
// quad's position comes from the cell index, so instances don't carry
// screen coordinates.
#version 460
#extension GL_GOOGLE_include_directive : require
#include "cell.glsl"

layout(location = 0) out vec2 local;              // pixel within the bitmap
layout(location = 1) flat out uvec2 atlas_origin;
layout(location = 2) flat out vec4 color;
layout(location = 3) flat out uint flags;

const vec2 CORNERS[6] = vec2[](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
    vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

void main() {
    Instance i = inst.instances[gl_InstanceIndex];
    uvec2 cell_index = uvec2(i.cell & 0xffffu, i.cell >> 16);
    // Sign-extend the 16-bit halves.
    ivec2 offset = ivec2(int(i.offset << 16) >> 16, int(i.offset) >> 16);
    vec2 size = vec2(i.size & 0xffffu, i.size >> 16);
    vec2 corner = CORNERS[gl_VertexIndex];

    vec2 top_left = vec2(origin + cell_index * cell) + vec2(offset);
    vec2 p = top_left + corner * size;
    gl_Position = vec4(p / vec2(screen) * 2.0 - 1.0, 0.0, 1.0);

    local = corner * size;
    atlas_origin = uvec2(i.atlas & 0xffffu, i.atlas >> 16);
    color = unpackUnorm4x8(i.color);
    flags = i.flags;
}
