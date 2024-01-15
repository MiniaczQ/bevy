#define_import_path surfels::view_bindings

#import bevy_render::view View

// View
@group(1) @binding(0) var<uniform> view: View;

// Surfel stack
const SURFEL_STACK_SIZE: u32 = 1024u;
const INVALID_SURFEL_ID: u32 = 4294967295u; // u32::MAX
@group(1) @binding(1) var<storage, read_write> unallocated_surfel_ids_stack: array<u32, SURFEL_STACK_SIZE>;
@group(1) @binding(2) var<storage, read_write> allocated_surfel_ids_stack: array<u32, SURFEL_STACK_SIZE>;
@group(1) @binding(3) var<storage, read_write> allocated_surfels_count: atomic<u32>;

// Surfel info
@group(1) @binding(4) var<storage, read_write> surfel_position: array<vec3<f32>, SURFEL_STACK_SIZE>;
@group(1) @binding(5) var<storage, read_write> surfel_normal: array<vec3<f32>, SURFEL_STACK_SIZE>;
@group(1) @binding(6) var<storage, read_write> surfel_irradiance: array<vec3<f32>, SURFEL_STACK_SIZE>;



// --- Stack operations ---
// SAFETY: stack can only run one operation type in parallel

fn allocate_surfel() -> u32 {
    let id = atomicAdd(&allocated_surfels_count, 1u);
    if id > SURFEL_STACK_SIZE {
        atomicSub(&allocated_surfels_count, 1u);
        return INVALID_SURFEL_ID;
    }
    allocated_surfel_ids_stack[id] = unallocated_surfel_ids_stack[SURFEL_STACK_SIZE - 1u - id];
    return id;
}

fn deallocate_surfel(id: u32) {
    atomicSub(&allocated_surfels_count, 1u);
    unallocated_surfel_ids_stack[SURFEL_STACK_SIZE - 1u - id] = allocated_surfel_ids_stack[id];
    surfel_position[id] = vec3<f32>(1.0 / 0.0); // +inf
}
