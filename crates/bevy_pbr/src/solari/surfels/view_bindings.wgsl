#define_import_path surfels::view_bindings

#import bevy_render::view View

// View
@group(1) @binding(0) var<uniform> view: View;

// Surfel stack
const MAX_SURFELS: u32 = 1024u;
const SURFEL_MAP_BITS: u32 = 32u;
const INVALID_SURFEL_ID: u32 = 4294967295u;

@group(1) @binding(1) var<storage, read_write> unallocated_surfel_ids_stack: array<u32, MAX_SURFELS>;
@group(1) @binding(2) var<storage, read_write> allocated_surfels_bitmap: array<u32, SURFEL_MAP_BITS>;
@group(1) @binding(3) var<storage, read_write> allocated_surfel_ids_count: atomic<u32>;

// Surfel info
@group(1) @binding(4) var<storage, read_write> surfel_position: array<vec4<f32>, MAX_SURFELS>;
@group(1) @binding(5) var<storage, read_write> surfel_normal: array<vec4<f32>, MAX_SURFELS>;
@group(1) @binding(6) var<storage, read_write> surfel_irradiance: array<vec4<f32>, MAX_SURFELS>;



// --- Stack operations ---
// SAFETY: stack can only run one operation type in parallel

// Pops unallocated stack and enables surfel in the map
fn allocate_surfel() -> u32 {
    let idx = atomicAdd(&allocated_surfel_ids_count, 1u); // pre operation
    if idx >= MAX_SURFELS {
        // Exceeded stack size, abort allocation
        atomicSub(&allocated_surfel_ids_count, 1u);
        return INVALID_SURFEL_ID;
    }
    let id = unallocated_surfel_ids_stack[idx];

    let bin = id / 32u;
    let bit = id % 32u;
    allocated_surfels_bitmap[bin] = allocated_surfels_bitmap[bin] | (1u << bit);
    return id;
}

// Pushes to unallocated stack and disables surfel in the map
// SAFETY: Access specific bin only by only one workgroup
fn deallocate_surfel(id: u32) {
    let idx = atomicSub(&allocated_surfel_ids_count, 1u) - 1u; // post operation
    unallocated_surfel_ids_stack[idx] = id;

    let bin = id / 32u;
    let bit = id % 32u;
    allocated_surfels_bitmap[bin] = allocated_surfels_bitmap[bin] & ~(1u << bit);
}
