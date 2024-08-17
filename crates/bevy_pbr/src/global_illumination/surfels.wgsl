#import bevy_render::view::View
#import bevy_render::globals::Globals
#import bevy_pbr::utils::{PI, rand_f, rand_vec2f, hsv2rgb, octahedral_decode, rand_range_u}
#import bevy_pbr::pbr_deferred_types::unpack_24bit_normal
#import bevy_pbr::global_illumination::bindings::{trace_ray, resolve_ray_hit, uv_depth_to_world, light_sources, first_hit_ray_trace, LightSample, sample_directional_light, sample_emissive_triangle, trace_directional_light, trace_emissive_triangle, LIGHT_SOURCE_DIRECTIONAL}
#import bevy_core_pipeline::tonemapping::tonemapping_luminance

struct SurfelSurface {
    position: vec3<f32>,
    normal: vec3<f32>,
    color: vec3<f32>,
}

struct SurfelIrradiance {
    mean: vec3<f32>,
    probes: u32,
    mean_squared: vec3<f32>,
    previous_sampled_light_id: u32,
}

struct Reservoir {
    light_id: u32,
    light_rng: u32,
    light_weight: f32,
    weight_sum: f32,
    sample_count: u32
}

struct SurfelCacheCell {
    count: u32,
    // Surfels contained withing some kernel.
    // For allocations this is 1x1. (8 on average)
    // For pixel lookup this is 5x5. (200 on average)
    ids: array<u32, 256>,
}

struct SurfelAllocationContext {
    allocations_left: f32,
}

// `SurfelCacheCell::ids` size minus one.
const SURFEL_CACHE_MAX_IDX: u32 = 255u;

const SPAWN_IF_LESS: i32 = 7;
const DESPAWN_IF_MORE: i32 = 9;

// Max amount of surfels. Has to be the same value as Rust-side shader code.
const MAX_SURFELS: u32 = 2048u;
// Size of surfel bitmap. [MAX_SURFELS / 32]
const SURFEL_MAP_BITS: u32 = 64u;

// How many lights (and surfels) are sampled by each surfel each frame.
const LIGHT_SAMPLES: u32 = 64u;
// Chance of sampling a surfel instead of an actual light source.
const SAMPLE_SURFEL_CHANCE: f32 = 0.5;
// Probability distribution function sample of a surfel. Cannot use 0.0 because of singularity.
const SURFEL_PDF: f32 = 10.0;

// How many samples get averaged by each surfel over time. Higher values mean slower changes, but less flickering.
const SURFEL_AVG_PROBES: u32 = 128u;

// Range at which surfels light surrounding pixels.
const AFFECTION_RANGE: f32 = 0.01; //0.075;
// Size of surfels in the debug view.
const DEBUG_SURFEL_SIZE: f32 = 0.0075;

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

// Screen space cache with 16x16 cells, each cell collects data in a 3x3 cell zone
@group(2) @binding(9) var<storage, read_write> surfel_cache: array<array<SurfelCacheCell, 16>, 16>;

// Surfel usage count, used as a metric to delete surfels
#ifdef ATOMIC_USAGE
@group(2) @binding(10) var<storage, read_write> surfel_usage: array<atomic<u32>, MAX_SURFELS>;
#else
@group(2) @binding(10) var<storage, read_write> surfel_usage: array<u32, MAX_SURFELS>;
#endif

@group(2) @binding(11) var diffuse_output: texture_storage_2d<rgba16float, read_write>;

// Buffer for indirect dispatch.
@group(2) @binding(12) var<storage, read_write> surfel_allocation_context: SurfelAllocationContext;

/// Store surfel occupancy in a 1x1 kernel.
@compute @workgroup_size(16, 16)
fn cache_surfels_1x1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cache_xy = global_id.xy;
    let cache_xy_f32 = vec2<f32>(cache_xy);
    let cell_min = (cache_xy_f32 - 8.0) / 8.0;
    let cell_max = (cache_xy_f32 - 7.0) / 8.0;

    var count = 0u;
    for (var id = 0u; id < MAX_SURFELS; id++) {
        let is_active = (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) != 0;

        let surfel_surface = surfels_surface[id];
        let ndc = world_to_ndc(surfel_surface.position);
        let is_within_cell = cell_min.x < ndc.x && ndc.x <= cell_max.x && cell_min.y < ndc.y && ndc.y <= cell_max.y;

        let is_correct = is_active && is_within_cell;

        let prev_id = surfel_cache[cache_xy.x][cache_xy.y].ids[count];
        surfel_cache[cache_xy.x][cache_xy.y].ids[count] = select(prev_id, id, is_correct);
        count = min(count + u32(is_correct), SURFEL_CACHE_MAX_IDX);
    }

    surfel_cache[cache_xy.x][cache_xy.y].count = count;
}

#ifdef ATOMIC_BITMAP
/// Deallocates one specific surfel.
///
/// SAFETY: Only call if the ID is allocated.
fn deallocate_surfel(id: u32) {
    let idx = atomicAdd(&unallocated_surfels, 1u);
    unallocated_surfel_ids_stack[idx] = id;

    let bin = id / 32u;
    let bit = id % 32u;
    atomicAnd(&allocated_surfels_bitmap[bin], ~(1u << bit));
}

// Attempts to despawn surfels that are too dense.
@compute @workgroup_size(16, 16)
fn despawn_surfels_high_density(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cache_xy = global_id.xy;
    var rng = globals.frame_count * 256 + global_id.x * 16 + global_id.y;

    var count = surfel_cache[cache_xy.x][cache_xy.y].count;
    let try_deallocate_n = i32(count) - DESPAWN_IF_MORE;

    for (var i = 0; i < try_deallocate_n; i++) {
        let removed_idx = rand_range_u(count, &rng);
        let removed_id = surfel_cache[cache_xy.x][cache_xy.y].ids[removed_idx];
        count -= 1u;
        surfel_cache[cache_xy.x][cache_xy.y].ids[removed_idx] = surfel_cache[cache_xy.x][cache_xy.y].ids[count];
        deallocate_surfel(removed_id);
    }
}

// Attempts to despawn surfels that don't contribute much to the final image.
@compute @workgroup_size(1)
fn despawn_surfels_low_usage(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var id = global_id.x;
    if (atomicLoad(&allocated_surfels_bitmap[id / 32u]) & (1u << (id % 32u))) == 0u {
        // Not active.
        return;
    }
    if surfel_usage[id] <= 0u {
        deallocate_surfel(id);
    }
    surfel_usage[id] = 0u;
}
#endif // ATOMIC_BITMAP

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

var<workgroup> allocations_left: atomic<i32>;

/// Attempts to spawn surfels to meet the average per cell.
@compute @workgroup_size(16, 16)
fn spawn_surfels(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if all(global_id.xy == vec2(0u)) {
        atomicStore(&allocations_left, i32(atomicLoad(&unallocated_surfels)));
    }
    workgroupBarrier();

    let cache_xy = global_id.xy;
    var rng = globals.frame_count * 256 + global_id.x * 16 + global_id.y;

    let count = surfel_cache[cache_xy.x][cache_xy.y].count;
    let try_allocate_n = SPAWN_IF_LESS - i32(count);

    for (var i = 0; i < try_allocate_n; i++) {
        let cell_uv = rand_vec2f(&rng);
        let pixel_uv = (vec2<f32>(cache_xy.xy) + cell_uv) / 16.0;
        let pixel_pos = vec2<u32>(pixel_uv * view.viewport.zw);
        let depth = textureLoad(depth_buffer, pixel_pos, 0i);
        if(depth == 0.0) { continue; } // Miss

        let nth_allocation = atomicSub(&allocations_left, 1);
        if nth_allocation <= 0 {
            // Ran out of IDs.
            return;
        }

        let world_xyz = uv_depth_to_world(depth, pixel_uv);
        let id = allocate_surfel();
        let gpixel = textureLoad(gbuffer, pixel_pos, 0i);
        let packed_normal = unpack_24bit_normal(gpixel.a);
        let world_normal = octahedral_decode(packed_normal);
        let base_color = pow(unpack4x8unorm(gpixel.r).rgb, vec3(2.2));
        surfels_surface[id] = SurfelSurface(world_xyz, world_normal, base_color);
        surfels_irradiance[id] = SurfelIrradiance(vec3(0.0), 0u, vec3(0.0), 0u);
    }
}
#endif // ATOMIC_BITMAP

/// Store surfel occupancy in a 5x5 kernel.
@compute @workgroup_size(16, 16)
fn cache_surfels_5x5(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cache_xy = global_id.xy;
    let cache_xy_f32 = vec2<f32>(cache_xy);
    let cell_min = (cache_xy_f32 - 10.0) / 8.0;
    let cell_max = (cache_xy_f32 - 5.0) / 8.0;

    var count = 0u;
    for (var id = 0u; id < MAX_SURFELS; id++) {
        let is_active = (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) != 0;

        let surfel_surface = surfels_surface[id];
        let ndc = world_to_ndc(surfel_surface.position);
        let is_within_cell = cell_min.x < ndc.x && ndc.x <= cell_max.x && cell_min.y < ndc.y && ndc.y <= cell_max.y;

        let is_correct = is_active && is_within_cell;

        let prev_id = surfel_cache[cache_xy.x][cache_xy.y].ids[count];
        surfel_cache[cache_xy.x][cache_xy.y].ids[count] = select(prev_id, id, is_correct);
        count = min(count + u32(is_correct), SURFEL_CACHE_MAX_IDX);
    }

    surfel_cache[cache_xy.x][cache_xy.y].count = count;
}

fn update_one_surfel(surfel: ptr<function, SurfelIrradiance>, irradiance: vec3<f32>, sampled_light_id: u32) {
    // Welford's online algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    (*surfel).probes = min((*surfel).probes + 1u, SURFEL_AVG_PROBES);
    let delta = irradiance - (*surfel).mean;
    (*surfel).mean += delta / f32((*surfel).probes);
    let delta2 = irradiance - (*surfel).mean;
    (*surfel).mean_squared += delta * delta2;

    (*surfel).previous_sampled_light_id = sampled_light_id;
}

fn update_reservoir(reservoir: ptr<function, Reservoir>, light_id: u32, light_rng: u32, light_weight: f32, rng: ptr<function, u32>) {
    (*reservoir).weight_sum += light_weight;
    (*reservoir).sample_count += 1u;
    if rand_f(rng) < light_weight / (*reservoir).weight_sum {
        (*reservoir).light_id = light_id;
        (*reservoir).light_rng = light_rng;
    }
}

fn sample_surfel(surfel_id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> LightSample {
    let surfel_surface = surfels_surface[surfel_id];
    let surfel_irradiance = surfels_irradiance[surfel_id];
    let light_distance = distance(ray_origin, surfel_surface.position);
    let ray_direction = (surfel_surface.position - ray_origin) / light_distance;

    let diffusion = saturate(dot(ray_direction, origin_world_normal));
    let attenuation = 1 + light_distance * light_distance;
    let is_toward_origin = f32(dot(ray_direction, surfel_surface.normal) < 0.0);
    var irradiance = surfel_irradiance.mean * surfel_surface.color * diffusion / attenuation * is_toward_origin;

    return LightSample(irradiance, SURFEL_PDF);
}

fn sample_surfel_or_light_source(id: u32, light_count: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> LightSample {
    var sample: LightSample;

    if id < MAX_SURFELS {
        sample = sample_surfel(id, ray_origin, origin_world_normal);
    } else {
        let light = light_sources[id - MAX_SURFELS];
        if light.kind == LIGHT_SOURCE_DIRECTIONAL {
            sample = sample_directional_light(light.id, ray_origin, origin_world_normal, state);
        } else {
            sample = sample_emissive_triangle(light.id, light.kind, ray_origin, origin_world_normal, state);
        }
    }

    sample.pdf /= f32(light_count);
    return sample;
}

fn trace_surfel(surfel_id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>) -> vec3<f32> {
    let surfel_surface = surfels_surface[surfel_id];
    let surfel_irradiance = surfels_irradiance[surfel_id];
    let light_distance = distance(ray_origin, surfel_surface.position);
    let ray_direction = (surfel_surface.position - ray_origin) / light_distance;
    let ray_hit = first_hit_ray_trace(ray_origin, ray_direction);

    let diffusion = saturate(dot(ray_direction, origin_world_normal));
    let attenuation = 1 + light_distance * light_distance;
    let is_toward_origin = f32(dot(ray_direction, surfel_surface.normal) < 0.0);
    let is_visible = f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);
    var irradiance = surfel_irradiance.mean * surfel_surface.color * diffusion / attenuation * is_toward_origin * is_visible;

    return irradiance;
}

fn trace_surfel_or_light_source(id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    if id < MAX_SURFELS {
        return trace_surfel(id, ray_origin, origin_world_normal);;
    } else {
        let light = light_sources[id - MAX_SURFELS];
        if light.kind == LIGHT_SOURCE_DIRECTIONAL {
            return trace_directional_light(light.id, ray_origin, origin_world_normal, state);
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

    let ndc = world_to_ndc(surfel_surface.position);
    let cache_xy = ndc_to_cache(ndc.xy);

    let brdf = surfel_surface.color / PI;

    var reservoir = Reservoir(0u, 0u, 0.0, 0.0, 0u);
    let light_count = arrayLength(&light_sources);

    // Light source sampling
    for (var i = 0u; i < LIGHT_SAMPLES; i++) {
        let light_id = select(MAX_SURFELS + rand_range_u(light_count, &rng), rand_range_u(MAX_SURFELS, &rng), rand_f(&rng) < SAMPLE_SURFEL_CHANCE);
        sample_one_light(&reservoir, &rng, light_id, light_count, surfel_surface, brdf);
    }

    rng = reservoir.light_rng;
    var irradiance = trace_surfel_or_light_source(reservoir.light_id, surfel_surface.position, surfel_surface.normal, &rng);

    let target_pdf = tonemapping_luminance(irradiance * brdf);
    let w = reservoir.weight_sum / (target_pdf * f32(reservoir.sample_count));
    reservoir.light_weight = select(0.0, w, target_pdf > 0.0);

    irradiance *= reservoir.light_weight;
    
    update_one_surfel(&surfel_irradiance, irradiance, reservoir.light_id);
    surfels_irradiance[id] = surfel_irradiance;
}

fn sample_one_light(reservoir: ptr<function, Reservoir>, rng: ptr<function, u32>, light_id: u32, light_count: u32, surfel_surface: SurfelSurface, brdf: vec3<f32>) {
    let light_rng = *rng;
    let sample = sample_surfel_or_light_source(light_id, light_count + MAX_SURFELS, surfel_surface.position, surfel_surface.normal, rng);
    let target_pdf = tonemapping_luminance(sample.irradiance * brdf);
    let light_weight = target_pdf / sample.pdf;
    update_reservoir(reservoir, light_id, light_rng, light_weight, rng);
}

/// Applies surfel diffuse for each pixel on the screen.
#ifdef ATOMIC_USAGE
@compute @workgroup_size(8, 8)
fn apply_surfel_diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2<u32>(view.viewport.zw)) { return; } // Outside of view

    let depth = textureLoad(depth_buffer, global_id.xy, 0i);
    if depth == 0.0 { return; } // Miss

    let pixel_uv = (vec2<f32>(global_id.xy) + 0.5) / view.viewport.zw;
    let world_pos = uv_depth_to_world(depth, pixel_uv);
    let view_distance = distance(view.world_position, world_pos);
    let gpixel = textureLoad(gbuffer, global_id.xy, 0i);
    let base_color = pow(unpack4x8unorm(gpixel.r).rgb, vec3(2.2));
    
    let ndc = uv_to_ndc(pixel_uv);
    let cache_xy = ndc_to_cache(ndc);

    var total_diffuse = vec3(0.0);
    var total_weight = 0.0;

    for (var idx = 0u; idx < surfel_cache[cache_xy.x][cache_xy.y].count; idx++) {
        let id = surfel_cache[cache_xy.x][cache_xy.y].ids[idx];
        let surfel_surface = surfels_surface[id];
        let surfel_distance = distance(world_pos, surfel_surface.position);
        let diffuse = surfels_irradiance[id].mean;
        let weight = saturate(view_distance * AFFECTION_RANGE - surfel_distance);
        total_diffuse += diffuse * weight;
        total_weight += weight;
        atomicAdd(&surfel_usage[id], u32(weight > 0.0));
    }

    total_diffuse = select(vec3(0.0), total_diffuse / total_weight, total_weight > 0.0) * base_color;

    textureStore(diffuse_output, global_id.xy, vec4<f32>(total_diffuse, 1.0));
}
#endif // ATOMIC_USAGE

@compute @workgroup_size(8, 8)
fn debug_surfels_view(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2<u32>(view.viewport.zw)) { return; } // Outside of view

    let depth = textureLoad(depth_buffer, global_id.xy, 0i);
    if depth == 0.0 { return; } // Miss

    let pixel_uv = (vec2<f32>(global_id.xy) + 0.5) / view.viewport.zw;
    let pixel_pos = uv_depth_to_world(depth, pixel_uv);

    let view_distance = distance(view.world_position, pixel_pos);
    let radius = view_distance * DEBUG_SURFEL_SIZE;
    
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; }
        let surfel_pos = surfels_surface[id].position;
        if distance(pixel_pos, surfel_pos) < radius {
            let color = hsv2rgb(f32(id) / f32(MAX_SURFELS), 1.0, 0.5);
            textureStore(diffuse_output, global_id.xy, vec4<f32>(color, 1.0));
            return;
        }
    }

    textureStore(diffuse_output, global_id.xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
}

fn world_to_ndc(pos: vec3<f32>) -> vec3<f32> {
    let ndc_raw = view.view_proj * vec4(pos, 1.0);
    return vec3(ndc_raw.x, -ndc_raw.y, ndc_raw.z) / ndc_raw.w;
}

fn ndc_to_world(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let pos_raw = view.inverse_view_proj * vec4(ndc.x, -ndc.y, depth, 1.0);
    return pos_raw.xyz / pos_raw.w;
}

fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * 2.0 - 1.0;
}

fn uv_depth_to_world(depth: f32, uv: vec2<f32>) -> vec3<f32> {
    let ndc = uv_to_ndc(uv);
    return ndc_to_world(ndc.xy, depth);
}

fn ndc_to_cache(ndc: vec2<f32>) -> vec2<u32> {
    return min(vec2<u32>((ndc + 1.0) * 8.0), vec2(15u, 15u));
}
