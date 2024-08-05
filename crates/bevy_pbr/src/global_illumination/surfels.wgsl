#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_pbr::utils::{rand_vec2f, hsv2rgb, octahedral_decode}
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::global_illumination::bindings::{sample_cosine_hemisphere, trace_ray, resolve_ray_hit, depth_to_world_position}

struct SurfelSurface {
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
}

struct SurfelIrradiance {
    mean: vec3<f32>,
    mean_squared: vec3<f32>,
    probes: u32,
}

const MAX_SURFELS: u32 = 1024u;
const SURFEL_MAP_BITS: u32 = 32u;
const MAX_SPAWNS: u32 = 64u;

const SURFEL_AVG_PROBES: u32 = 32u;

const DEBUG_SURFEL_SIZE: f32 = 0.0078125;
const AFFECTION_RANGE: f32 = 0.1;

@group(2) @binding(0) var<uniform> view: View;
@group(2) @binding(1) var<uniform> globals: Globals;
@group(2) @binding(2) var depth_buffer: texture_depth_2d;
@group(2) @binding(3) var gbuffer: texture_2d<u32>;

// Stack with IDs of unallocated surfels. Allows for surfel allocation.
@group(2) @binding(4) var<storage, read_write> unallocated_surfel_ids_stack: array<u32, MAX_SURFELS>;
// Bitmap of which surfels are currently allocated. Allows for filtering active surfels.
#ifdef ATOMIC_BITMAP
@group(2) @binding(5) var<storage, read_write> allocated_surfels_bitmap: array<atomic<u32>, SURFEL_MAP_BITS>;
#else
@group(2) @binding(5) var<storage, read_write> allocated_surfels_bitmap: array<u32, SURFEL_MAP_BITS>;
#endif
// Stack pointer.
@group(2) @binding(6) var<storage, read_write> unallocated_surfels: atomic<u32>;

// Surfel info.
@group(2) @binding(7) var<storage, read_write> surfels_surface: array<SurfelSurface, MAX_SURFELS>;
@group(2) @binding(8) var<storage, read_write> surfels_irradiance: array<SurfelIrradiance, MAX_SURFELS>;

@group(2) @binding(9) var diffuse_output: texture_storage_2d<rgba16float, read_write>;

// Buffer for indirect dispatch.
#ifdef INDIRECT_ALLOCATE
@group(2) @binding(10) var<storage, read_write> surfels_to_allocate: vec3<u32>;
#endif

#ifdef INDIRECT_ALLOCATE
/// Stores unallocated surfel count into a buffer for spawning through indirect dispatch.
@compute @workgroup_size(1)
fn count_unallocated_surfels() {
    surfels_to_allocate.x = min(atomicLoad(&unallocated_surfels), MAX_SPAWNS);
    surfels_to_allocate.y = 1u;
    surfels_to_allocate.z = 1u;
}
#endif


#ifdef ATOMIC_BITMAP
/// Attempts to allocate a new surfel and returns it's ID.
///
/// SAFETY: Only call if there are enough available ID's.
fn allocate_surfel() -> u32 {
    let idx = atomicSub(&unallocated_surfels, 1u) - 1u;
    let id = unallocated_surfel_ids_stack[idx];

    let bin = id / 32u;
    let bit = id % 32u;
    atomicOr(&allocated_surfels_bitmap[bin], (1u << bit));
    return id;
}

/// Attempts to spawn a single surfel through a random raycast into the scene.
@compute @workgroup_size(1)
fn spawn_surfels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var rng = globals.frame_count * MAX_SPAWNS + global_id.x;
    let pixel_uv = rand_vec2f(&rng);
    let pixel_pos = vec2<u32>(pixel_uv * view.viewport.zw);
    let depth = textureLoad(depth_buffer, pixel_pos, 0i);
    if(depth == 0.0) { return; } // Miss

    let world_xyz = depth_to_world_position(depth, pixel_uv);
    let id = allocate_surfel();
    let gpixel = textureLoad(gbuffer, pixel_pos, 0i);
    let packed_normal = unpack_24bit_normal(gpixel.a);
    let world_normal = octahedral_decode(packed_normal);
    let base_color = pow(unpack4x8unorm(gpixel.r).rgb, vec3(2.2));
    surfels_surface[id] = SurfelSurface(world_xyz, world_normal, base_color);
    surfels_irradiance[id] = SurfelIrradiance(vec3(0.0), vec3<f32>(0.0), 0u);
}
#endif

// Welford's online algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
fn update_one_surfel(surfel: ptr<function, SurfelIrradiance>, irradiance: vec3<f32>) {
    (*surfel).probes = min((*surfel).probes + 1u, SURFEL_AVG_PROBES);
    let delta = irradiance - (*surfel).mean;
    (*surfel).mean += delta / f32((*surfel).probes);
    let delta2 = irradiance - (*surfel).mean;
    (*surfel).mean_squared += delta * delta2;
}

/// Updates the diffuse of each surfel.
@compute @workgroup_size(32)
fn update_surfels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var id = global_id.x;
    if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { return; } // Surfel not active

    let surfel_surface = surfels_surface[id];
    var surfel_irradiance = surfels_irradiance[id];
    var rng = globals.frame_count * MAX_SURFELS + global_id.x;
    var irradiance = vec3<f32>(1.0);

    //var rng2 = rng;
    
    update_one_surfel(&surfel_irradiance, irradiance);
    surfels_irradiance[id] = surfel_irradiance;
}

/// Applies surfel diffuse for each pixel on the screen.
@compute @workgroup_size(8, 8)
fn apply_surfel_diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2<u32>(view.viewport.zw)) { return; } // Outside of view

    let depth = textureLoad(depth_buffer, global_id.xy, 0i);
    if depth == 0.0 { return; } // Miss

    let pixel_uv = (vec2<f32>(global_id.xy) + 0.5) / view.viewport.zw;
    let world_pos = depth_to_world_position(depth, pixel_uv);
    let view_dis = distance(view.world_position, world_pos);
    
    var lighting = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

        let surfel_surface = surfels_surface[id];
        let distance = distance(world_pos, surfel_surface.position);
        let weight = max(view_dis * AFFECTION_RANGE - distance, 0.0);
        lighting += surfels_irradiance[id].mean * weight;
        total_weight += weight;
    }

    lighting = select(lighting, lighting / total_weight, total_weight > 0.0);

    textureStore(diffuse_output, global_id.xy, vec4<f32>(lighting, 1.0));
}

/// Deallocates one specific surfel.
///
/// SAFETY: Only call if the ID is allocated.
fn deallocate_surfel(id: u32) {
    let idx = atomicAdd(&unallocated_surfels, 1u);
    unallocated_surfel_ids_stack[idx] = id;

    let bin = id / 32u;
    let bit = id % 32u;
    allocated_surfels_bitmap[bin] = allocated_surfels_bitmap[bin] & ~(1u << bit);
}

// Attempts to despawn all surfels outside of view frustum.
@compute @workgroup_size(32)
fn despawn_surfels(@builtin(local_invocation_index) local_idx: u32) {
    var id = local_idx * 32u;
    let max_id = id + MAX_SURFELS / 32u;
    for (; id < max_id; id++) {
        // Possibly move to other condition
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u {
            // Surfel not active
            continue;
        }
        let ndc_raw = view.view_proj * vec4(surfels_surface[id].position, 1.0);
        let ndc = ndc_raw.xyz / ndc_raw.w;
        if ndc.x < -1.0 || 1.0 < ndc.x || ndc.y < -1.0 || 1.0 < ndc.y || ndc.z < -1.0 || 1.0 < ndc.z {
            deallocate_surfel(id);
        }
    }
}

@compute @workgroup_size(8, 8)
fn debug_surfels_view(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2<u32>(view.viewport.zw)) { return; } // Outside of view

    let depth = textureLoad(depth_buffer, global_id.xy, 0i);
    if depth == 0.0 { return; } // Miss

    let pixel_uv = (vec2<f32>(global_id.xy) + 0.5) / view.viewport.zw;
    let pixel_pos = depth_to_world_position(depth, pixel_uv);

    let view_distance = distance(view.world_position, pixel_pos);
    let radius = view_distance * DEBUG_SURFEL_SIZE;
    
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active
        let surfel_pos = surfels_surface[id].position;
        if distance(pixel_pos, surfel_pos) < radius {
            let color = hsv2rgb(f32(id) / f32(MAX_SURFELS), 1.0, 0.5);
            textureStore(diffuse_output, global_id.xy, vec4<f32>(color, 1.0));
            return;
        }
    }

    textureStore(diffuse_output, global_id.xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
}

fn depth_to_world_position(depth: f32, uv: vec2<f32>) -> vec3<f32> {
    let ndc = (uv - 0.5) * vec2(2.0, -2.0); // TODO: why is Y negated???
    let point = view.inverse_view_proj * vec4(ndc, depth, 1.0);
    return point.xyz / point.w;
}
