// Shared by cell.vert and cell.frag. Layouts must match src/text.rs.

#extension GL_EXT_buffer_reference : require

const uint FLAG_SOLID = 1u; // fill the quad instead of reading the atlas

// One glyph (or solid block) drawn in one cell. Packed 16-bit pairs keep it
// at 24 bytes.
struct Instance {
    uint cell;    // column | row << 16
    uint offset;  // bitmap top-left relative to the cell's: x | y << 16, signed
    uint atlas;   // atlas origin: x | y << 16
    uint size;    // bitmap size: width | height << 16
    uint color;   // RGBA8, straight alpha
    uint flags;
};

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer Instances {
    Instance instances[];
};

layout(push_constant) uniform Push {
    Instances inst;
    uvec2 screen;  // target size in pixels
    uvec2 cell;    // cell size in pixels
    uvec2 origin;  // top-left of the grid in pixels
};
