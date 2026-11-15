// Shared by deposit.comp and resolve.comp: the GPU glyph rasterizer's data.
// Layouts must match src/raster/mod.rs.

#extension GL_EXT_buffer_reference : require

// Fixed-point scale for accumulated area (no float atomics in core Vulkan).
const float SCALE = 65536.0;

// A line segment of a glyph outline, in that glyph's bitmap pixels (y down).
struct Segment {
    vec2 p0;
    vec2 p1;
    uint job;
    uint _pad;
};

// One glyph to rasterize.
struct Job {
    uvec2 atlas_origin;
    uvec2 size;          // bitmap size, padding included
    uint accum_offset;   // first cell of this glyph in the accumulation grid
    uint _pad;
};

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Segments {
    Segment segments[];
};

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer Jobs {
    Job jobs[];
};

// size.y rows of (size.x + 2) cells per job: one spare column on the right
// for area spilling past the last pixel, one so the row sum can end on it.
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Accum {
    int cells[];
};

layout(push_constant) uniform Push {
    Segments seg;
    Jobs job;
    Accum accum;
    uint segment_count;
    uint job_base;   // resolve: first job of this dispatch
};
