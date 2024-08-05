#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_pbr::utils::{rand_vec2f, hsv2rgb, octahedral_decode}
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::global_illumination::bindings::{sample_cosine_hemisphere, trace_ray, resolve_ray_hit, depth_to_world_position}

struct SurfelIrradiance {
    mean: vec3<f32>,
    mean_squared: vec3<f32>,
    probes: u32,
}

const MAX_SURFELS: u32 = 1024u;
const SURFEL_MAP_BITS: u32 = 32u;
const MAX_SPAWNS: u32 = 64u;

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
@group(2) @binding(7) var<storage, read_write> surfel_position: array<vec4<f32>, MAX_SURFELS>;
@group(2) @binding(8) var<storage, read_write> surfel_normal: array<vec4<f32>, MAX_SURFELS>;
@group(2) @binding(9) var<storage, read_write> surfel_irradiance: array<SurfelIrradiance, MAX_SURFELS>;

@group(2) @binding(10) var diffuse_output: texture_storage_2d<rgba16float, read_write>;

// Buffer for indirect dispatch.
#ifdef INDIRECT_ALLOCATE
@group(2) @binding(11) var<storage, read_write> surfels_to_allocate: vec3<u32>;
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
    surfel_position[id] = vec4<f32>(world_xyz, 1.0);
    let gpixel = textureLoad(gbuffer, pixel_pos, 0i);
    let packed_normal = unpack_24bit_normal(gpixel.a);
    let world_normal = octahedral_decode(packed_normal);
    surfel_normal[id] = vec4<f32>(world_normal, 1.0);
    surfel_irradiance[id] = SurfelIrradiance(vec3(0.0), vec3<f32>(0.0), 0u);
}
#endif

/// Updates the diffuse of each surfel.
@compute @workgroup_size(64)
fn update_surfels(@builtin(local_invocation_index) local_idx: u32) {
    var id = local_idx * 32u;
    let max_id = id + MAX_SURFELS / 32u;
    for (; id < max_id; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

        let surfel_pos = surfel_position[id].xyz;
        let surfel_nor = surfel_normal[id].xyz;
        var surfel_irr = surfel_irradiance[id];
        var rng = globals.frame_count * MAX_SURFELS + local_idx;
        var irradiance = vec3<f32>(0.0);

        let ray_dir = sample_cosine_hemisphere(surfel_nor, &rng);
        let ray_hit = trace_ray(surfel_pos, ray_dir, 0.001, 100.0);
        if ray_hit.kind != RAY_QUERY_INTERSECTION_NONE {
            let hit_dis = 0.001 + ray_hit.t * 100.0;
            let hit_pos = surfel_pos + ray_dir * hit_dis;
            let solari_ray_hit = resolve_ray_hit(ray_hit);

            var lighting = vec3<f32>(0.0);

            //let direct_lighting_diffuse = sample_direct_lighting_diffuse(hit_pos + solari_ray_hit.world_normal * 0.001, solari_ray_hit.world_normal, &rng);

            let direct_lighting_diffuse = vec3(1.0f);

            var total_weight = 0.000001;
            for (var id = 0u; id < MAX_SURFELS; id++) {
                if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

                let surfel_pos2 = surfel_position[id].xyz;
                let surfel_dis2 = distance(hit_pos, surfel_pos2);
                let weight = max(1.0 - surfel_dis2, 0.0);
                lighting += surfel_irradiance[id].mean * weight;
                total_weight += weight;
            }

            let surfel_illu = lighting / total_weight;
            irradiance += solari_ray_hit.material.base_color * (direct_lighting_diffuse + surfel_illu);
        }

        // Welford's online algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        surfel_irr.probes = min(surfel_irr.probes + 1u, 32u);
        let delta = irradiance - surfel_irr.mean;
        surfel_irr.mean += delta / f32(surfel_irr.probes);
        let delta2 = irradiance - surfel_irr.mean;
        surfel_irr.mean_squared += delta * delta2;
        
        surfel_irradiance[id] = surfel_irr;
    }
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

        let surfel_pos = surfel_position[id].xyz;
        let surfel_dis = distance(world_pos, surfel_pos);
        let weight = max(view_dis * AFFECTION_RANGE - surfel_dis, 0.0);
        lighting += surfel_irradiance[id].mean * weight;
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
        let world_surfel_pos = surfel_position[id];
        let ss_surfel_pos = view.view_proj * world_surfel_pos;
        let pos = ss_surfel_pos.xyz / ss_surfel_pos.w;
        if pos.x < -1.0 || 1.0 < pos.x || pos.y < -1.0 || 1.0 < pos.y || pos.z < -1.0 || 1.0 < pos.z {
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
    let world_pos = depth_to_world_position(depth, pixel_uv);

    let view_distance = distance(view.world_position, world_pos);
    let radius = view_distance * DEBUG_SURFEL_SIZE;
    
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active
        let surfel_pos = surfel_position[id].xyz;
        if distance(world_pos, surfel_pos) < radius {
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
