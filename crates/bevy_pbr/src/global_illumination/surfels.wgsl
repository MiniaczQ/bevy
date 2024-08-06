#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_pbr::utils::{PI, rand_f, rand_vec2f, hsv2rgb, octahedral_decode, rand_range_u}
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::global_illumination::bindings::{trace_ray, resolve_ray_hit, depth_to_world_position, light_sources, first_hit_ray_trace, LightSample, sample_directional_light, sample_emissive_triangle, trace_directional_light, trace_emissive_triangle, LIGHT_SOURCE_DIRECTIONAL}
#import bevy_core_pipeline::tonemapping::tonemapping_luminance

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

struct Reservoir {
    light_id: u32,
    light_rng: u32,
    light_weight: f32,
    weight_sum: f32,
    sample_count: u32
}

// Max amount of surfels. Has to be the same value as Rust-side shader code.
const MAX_SURFELS: u32 = 1024u;
// Size of surfel bitmap. [MAX_SURFELS / 32]
const SURFEL_MAP_BITS: u32 = 32u;
// How many surfels can be spawned each frame at most.
const MAX_SPAWNS: u32 = 32u;

// How many lights (and surfels) are sampled by each surfel each frame.
const LIGHT_SAMPLES: u32 = 128u;
// Chance of sampling a surfel instead of an actual light source.
const SAMPLE_SURFEL_CHANCE: f32 = 0.5;
// Probability distribution sample of a surfel. Cannot use 0.0 because of singularity.
const SURFEL_PDF: f32 = 100.0;

// How many samples get averaged by each surfel over time. Higher values mean slower changes, but less flickering.
const SURFEL_AVG_PROBES: u32 = 128u;

// Range at which surfels light surrounding pixels.
const AFFECTION_RANGE: f32 = 0.2;
// Size of surfels in the debug view.
const DEBUG_SURFEL_SIZE: f32 = 0.0078125;

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

fn update_reservoir(reservoir: ptr<function, Reservoir>, light_id: u32, light_rng: u32, light_weight: f32, rng: ptr<function, u32>) {
    (*reservoir).weight_sum += light_weight;
    (*reservoir).sample_count += 1u;
    if rand_f(rng) < light_weight / (*reservoir).weight_sum {
        (*reservoir).light_id = light_id;
        (*reservoir).light_rng = light_rng;
    }
}

fn sample_surfel_or_light_source(id: u32, light_count: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> LightSample {
    var sample: LightSample;

    if id < MAX_SURFELS {
        let surfel_id = id;
        let surfel_surface = surfels_surface[surfel_id];
        let surfel_irradiance = surfels_irradiance[surfel_id];
        let ray_direction = surfel_surface.position - ray_origin;
        let distance = length(ray_direction);
        var irradiance = surfel_surface.color * surfel_irradiance.mean;
        irradiance = irradiance / (1 + distance * distance);
        sample = LightSample(irradiance, SURFEL_PDF);
    } else {
        let light_id = id - MAX_SURFELS;
        let light = light_sources[light_id];
        if light.kind == LIGHT_SOURCE_DIRECTIONAL {
            sample = sample_directional_light(light.id, ray_origin, state);
        } else {
            sample = sample_emissive_triangle(light.id, light.kind, ray_origin, origin_world_normal, state);
        }
    }

    sample.pdf /= f32(light_count);
    return sample;
}

fn trace_surfel_or_light_source(id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    // Surfels are virtual lights.
    if id < MAX_SURFELS {
        let surfel_id = id;
        let surfel_surface = surfels_surface[surfel_id];
        let surfel_irradiance = surfels_irradiance[surfel_id];
        let ray_direction = surfel_surface.position - ray_origin;
        let ray_hit = first_hit_ray_trace(ray_origin, ray_direction);
        let light_visible = f32(dot(ray_direction, surfel_surface.normal) < 0.0 && ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
        let distance = length(ray_direction);
        var irradiance = surfel_surface.color * surfel_irradiance.mean;
        irradiance = irradiance / (1 + distance * distance);
        return irradiance * light_visible;
    } else {
        let light_id = id - MAX_SURFELS;
        let light = light_sources[light_id];
        if light.kind == LIGHT_SOURCE_DIRECTIONAL {
            return trace_directional_light(light.id, ray_origin, state);
        } else {
            return trace_emissive_triangle(light.id, light.kind, ray_origin, origin_world_normal, state);
        }
    }
}

/// Updates the diffuse of each surfel.
@compute @workgroup_size(32)
fn update_surfels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var id = global_id.x;
    if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { return; } // Surfel not active

    let surfel_surface = surfels_surface[id];
    var surfel_irradiance = surfels_irradiance[id];
    var rng = globals.frame_count * MAX_SURFELS + global_id.x;

    let brdf = surfel_surface.color / PI;

    var reservoir = Reservoir(0u, 0u, 0.0, 0.0, 0u);
    let light_count = arrayLength(&light_sources);
    for (var i = 0u; i < LIGHT_SAMPLES; i++) {
        let light_id = select(MAX_SURFELS + rand_range_u(light_count, &rng), rand_range_u(MAX_SURFELS, &rng), rand_f(&rng) < SAMPLE_SURFEL_CHANCE);
        let light_rng = rng;
        let sample = sample_surfel_or_light_source(light_id, light_count + MAX_SURFELS, surfel_surface.position, surfel_surface.normal, &rng);
        let target_pdf = tonemapping_luminance(sample.irradiance * brdf);
        let light_weight = target_pdf / sample.pdf;

        update_reservoir(&reservoir, light_id, light_rng, light_weight, &rng);
    }

    rng = reservoir.light_rng;
    var irradiance = trace_surfel_or_light_source(reservoir.light_id, surfel_surface.position, surfel_surface.normal, &rng);

    let target_pdf = tonemapping_luminance(irradiance * brdf);
    let w = reservoir.weight_sum / (target_pdf * f32(reservoir.sample_count));
    reservoir.light_weight = select(0.0, w, target_pdf > 0.0);

    irradiance *= reservoir.light_weight;
    
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
    let view_distance = distance(view.world_position, world_pos);
    let gpixel = textureLoad(gbuffer, global_id.xy, 0i);
    let base_color = pow(unpack4x8unorm(gpixel.r).rgb, vec3(2.2));
    
    var lighting = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

        let surfel_surface = surfels_surface[id];
        let surfel_distance = distance(world_pos, surfel_surface.position);
        let weight = max(view_distance * AFFECTION_RANGE - surfel_distance, 0.0);
        lighting += surfels_irradiance[id].mean * weight * surfel_surface.color;
        total_weight += weight;
    }

    lighting = select(lighting, lighting / total_weight, total_weight > 0.0);
    lighting *= base_color;

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
