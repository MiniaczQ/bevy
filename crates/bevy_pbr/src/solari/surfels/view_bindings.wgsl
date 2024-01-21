#define_import_path surfels::view_bindings

#import bevy_render::view View

struct SurfelIrradiance {
    mean: vec3<f32>,
    mean_squared: vec3<f32>,
    probes: u32,
}

// Input
@group(1) @binding(0) var<uniform> view: View;
@group(1) @binding(1) var depth_buffer: texture_depth_2d;
@group(1) @binding(2) var normals_buffer: texture_2d<f32>;

// Surfel stack
const MAX_SURFELS: u32 = 1024u;
const SURFEL_MAP_BITS: u32 = 32u;
const INVALID_SURFEL_ID: u32 = 4294967295u;
const MAX_SPAWNS: u32 = 64u;

@group(1) @binding(3) var<storage, read_write> unallocated_surfel_ids_stack: array<u32, MAX_SURFELS>;
#ifdef ATOMIC_SURFEL_MAP_ENABLED
@group(1) @binding(4) var<storage, read_write> allocated_surfels_bitmap: array<atomic<u32>, SURFEL_MAP_BITS>;
#else
@group(1) @binding(4) var<storage, read_write> allocated_surfels_bitmap: array<u32, SURFEL_MAP_BITS>;
#endif
@group(1) @binding(5) var<storage, read_write> unallocated_surfels: atomic<u32>;

// Surfel info
@group(1) @binding(6) var<storage, read_write> surfel_position: array<vec4<f32>, MAX_SURFELS>;
@group(1) @binding(7) var<storage, read_write> surfel_normal: array<vec4<f32>, MAX_SURFELS>;
@group(1) @binding(8) var<storage, read_write> surfel_irradiance: array<SurfelIrradiance, MAX_SURFELS>;

// Output
@group(1) @binding(9) var diffuse_irradiance_output: texture_storage_2d<rgba16float, write>;

@group(1) @binding(10) var<storage, read_write> surfel_grid_allocate: array<array<u32, 16>, 16>;
#ifdef SURFELS_TO_ALLOCATE_ENABLED
@group(1) @binding(11) var<storage, read_write> surfels_to_allocate: vec3<u32>;
#endif
