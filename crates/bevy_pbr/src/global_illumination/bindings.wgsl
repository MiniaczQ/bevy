#define_import_path bevy_pbr::global_illumination::bindings

#import bevy_pbr::utils::{PI, rand_f, rand_vec2f, rand_range_u}

struct Material {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    base_color_texture_id: u32,
    normal_map_texture_id: u32,
    emissive_texture_id: u32,
    _padding: u32,
}

const TEXTURE_MAP_NONE = 0xFFFFFFFFu;

struct ResolvedMaterial {
    base_color: vec3<f32>,
    emissive: vec3<f32>,
}

struct ResolvedRayHit {
    world_position: vec3<f32>,
    world_normal: vec3<f32>,
    geometric_world_normal: vec3<f32>,
    uv: vec2<f32>,
    material: ResolvedMaterial,
    triangle_area: f32,
}

struct LightSource {
    kind: u32,
    id: u32,
}

const LIGHT_SOURCE_DIRECTIONAL = 0xFFFFFFFFu;

struct DirectionalLight {
    direction_to_light: vec3<f32>,
    color: vec4<f32>,
}

struct PackedVertex {
    a: vec4<f32>,
    b: vec4<f32>,
    tangent: vec4<f32>,
}

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tangent: vec4<f32>,
}

struct VertexBuffer { vertices: array<PackedVertex> }

struct IndexBuffer { indices: array<u32> }

@group(0) @binding(0) var<storage> vertex_buffers: binding_array<VertexBuffer>;
@group(0) @binding(1) var<storage> index_buffers: binding_array<IndexBuffer>;
@group(0) @binding(2) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(3) var samplers: binding_array<sampler>;

@group(1) @binding(0) var tlas: acceleration_structure;
@group(1) @binding(1) var<storage> mesh_material_ids: array<u32>;
@group(1) @binding(2) var<storage> transforms: array<mat4x4<f32>>;
@group(1) @binding(3) var<storage> materials: array<Material>;
@group(1) @binding(4) var<storage> light_sources: array<LightSource>;
@group(1) @binding(5) var<storage> directional_lights: array<DirectionalLight>;

const RAY_T_MIN = 0.001;
const RAY_T_MAX = 100000.0;
const RAY_NO_CULL = 0xFFu;

fn trace_ray(ray_origin: vec3<f32>, ray_direction: vec3<f32>, ray_t_min: f32, ray_t_max: f32) -> RayIntersection {
    let ray = RayDesc(RAY_FLAG_NONE, RAY_NO_CULL, ray_t_min, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq);
}

fn unpack_vertex(packed: PackedVertex) -> Vertex {
    var vertex: Vertex;
    vertex.position = packed.a.xyz;
    vertex.normal = vec3(packed.a.w, packed.b.xy);
    vertex.uv = packed.b.zw;
    vertex.tangent = packed.tangent;
    return vertex;
}

fn sample_texture(id: u32, uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(textures[id], samplers[id], uv, 0.0).rgb;
}

fn resolve_material(material: Material, uv: vec2<f32>) -> ResolvedMaterial {
    var m: ResolvedMaterial;

    m.base_color = material.base_color.rgb;
    if material.base_color_texture_id != TEXTURE_MAP_NONE {
        m.base_color *= sample_texture(material.base_color_texture_id, uv);
    }

    m.emissive = material.emissive.rgb;
    if material.emissive_texture_id != TEXTURE_MAP_NONE {
        m.emissive *= sample_texture(material.emissive_texture_id, uv);
    }

    return m;
}

fn resolve_ray_hit_inner(object_id: u32, triangle_id: u32, barycentrics_input: vec2<f32>) -> ResolvedRayHit {
    let mm_ids = mesh_material_ids[object_id];
    let mesh_id = mm_ids >> 16u;
    let material_id = mm_ids & 0xFFFFu;

    let index_buffer = &index_buffers[mesh_id].indices;
    let vertex_buffer = &vertex_buffers[mesh_id].vertices;
    let material = materials[material_id];

    let indices_i = (triangle_id * 3u) + vec3(0u, 1u, 2u);
    let indices = vec3((*index_buffer)[indices_i.x], (*index_buffer)[indices_i.y], (*index_buffer)[indices_i.z]);
    let vertices = array<Vertex, 3>(unpack_vertex((*vertex_buffer)[indices.x]), unpack_vertex((*vertex_buffer)[indices.y]), unpack_vertex((*vertex_buffer)[indices.z]));
    let barycentrics = vec3(1.0 - barycentrics_input.x - barycentrics_input.y, barycentrics_input);

    let transform = transforms[object_id];
    let local_position = mat3x3(vertices[0].position, vertices[1].position, vertices[2].position) * barycentrics;
    let world_position = (transform * vec4(local_position, 1.0)).xyz;

    let uv = mat3x2(vertices[0].uv, vertices[1].uv, vertices[2].uv) * barycentrics;

    let local_normal = mat3x3(vertices[0].normal, vertices[1].normal, vertices[2].normal) * barycentrics;
    var world_normal = normalize(mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_normal);
    let geometric_world_normal = world_normal;
    if material.normal_map_texture_id != TEXTURE_MAP_NONE {
        let local_tangent = mat3x3(vertices[0].tangent.xyz, vertices[1].tangent.xyz, vertices[2].tangent.xyz) * barycentrics;
        let world_tangent = normalize(mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz) * local_tangent);
        let N = world_normal;
        let T = world_tangent;
        let B = vertices[0].tangent.w * cross(N, T);
        let Nt = sample_texture(material.normal_map_texture_id, uv);
        world_normal = normalize(Nt.x * T + Nt.y * B + Nt.z * N);
    }

    let resolved_material = resolve_material(material, uv);

    let triangle_edge0 = vertices[0].position - vertices[1].position;
    let triangle_edge1 = vertices[0].position - vertices[2].position;
    let triangle_area = length(cross(triangle_edge0, triangle_edge1)) / 2.0;

    return ResolvedRayHit(world_position, world_normal, geometric_world_normal, uv, resolved_material, triangle_area);
}

fn resolve_ray_hit(ray_hit: RayIntersection) -> ResolvedRayHit {
    return resolve_ray_hit_inner(ray_hit.instance_custom_index, ray_hit.primitive_index, ray_hit.barycentrics);
}

// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec28%3A303
fn sample_cosine_hemisphere(normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = 1.0 - 2.0 * rand_f(state);
    let phi = 2.0 * PI * rand_f(state);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let x = normal.x + sin_theta * cos(phi);
    let y = normal.y + sin_theta * sin(phi);
    let z = normal.z + cos_theta;
    return vec3(x, y, z);
}

// https://jcgt.org/published/0006/01/01/paper.pdf
fn generate_tbn(normal: vec3<f32>) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (sign + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = vec3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    let bitangent = vec3(b, sign + normal.y * normal.y * a, -normal.y);
    return mat3x3(tangent, bitangent, normal);
}

struct LightSample {
    irradiance: vec3<f32>,
    pdf: f32,
}

// https://en.wikipedia.org/wiki/Angular_diameter#Use_in_astronomy
// https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec30%3A305
fn sample_directional_light(id: u32, ray_origin: vec3<f32>, state: ptr<function, u32>) -> LightSample {
    let light = directional_lights[id];

    // Angular diameter of the sun projected onto a disk as viewed from earth = ~0.5 degrees
    // cos(0.25)
    let cos_theta_max = 0.99999048072;

    // 1 / (2 * PI * (1 - cos_theta_max))
    let pdf = 16719.2206859;

    let r = rand_vec2f(state);
    let cos_theta = (1.0 - r.x) + r.x * cos_theta_max;
    // TODO: is this necessary?
    //let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    //let phi = r.y * 2.0 * PI;
    //var ray_direction = vec3(vec2(cos(phi), sin(phi)) * sin_theta, cos_theta);
    //ray_direction = generate_tbn(light.direction_to_light) * ray_direction;

    let irradiance = light.color.rgb * cos_theta;

    return LightSample(irradiance, pdf);
}

fn trace_directional_light(id: u32, ray_origin: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let light = directional_lights[id];

    let cos_theta_max = 0.99999048072;

    let r = rand_vec2f(state);
    let cos_theta = (1.0 - r.x) + r.x * cos_theta_max;
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    let phi = r.y * 2.0 * PI;
    var ray_direction = vec3(vec2(cos(phi), sin(phi)) * sin_theta, cos_theta);
    ray_direction = generate_tbn(light.direction_to_light) * ray_direction;

    let ray = RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_NO_CULL, RAY_T_MIN, RAY_T_MAX, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    let ray_hit = rayQueryGetCommittedIntersection(&rq);
    let light_visible = f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);

    let irradiance = light.color.rgb * cos_theta;

    return irradiance * light_visible;
}

fn sample_emissive_triangle(object_id: u32, triangle_id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> LightSample {
    // https://www.realtimerendering.com/raytracinggems/unofficial_RayTracingGems_v1.9.pdf#0004286901.INDD%3ASec22%3A297
    var barycentrics = rand_vec2f(state);
    // TODO: faster?
    barycentrics = select(barycentrics, 1.0 - barycentrics, barycentrics.x + barycentrics.y > 1.0);
    //if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    let light_hit = resolve_ray_hit_inner(object_id, triangle_id, barycentrics);

    let pdf = 1.0 / light_hit.triangle_area;

    let light_distance = distance(ray_origin, light_hit.world_position);
    let ray_direction = (light_hit.world_position - ray_origin) / light_distance;
    let cos_theta_origin = saturate(dot(ray_direction, origin_world_normal));
    let cos_theta_light = saturate(dot(-ray_direction, light_hit.world_normal));
    let light_distance_squared = light_distance * light_distance;
    let irradiance = light_hit.material.emissive.rgb * cos_theta_origin * (cos_theta_light / light_distance_squared);

    return LightSample(irradiance, pdf);
}

fn trace_emissive_triangle(object_id: u32, triangle_id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    var barycentrics = rand_vec2f(state);
    // TODO: faster?
    barycentrics = select(barycentrics, 1.0 - barycentrics, barycentrics.x + barycentrics.y > 1.0);
    //if barycentrics.x + barycentrics.y > 1.0 { barycentrics = 1.0 - barycentrics; }
    let light_hit = resolve_ray_hit_inner(object_id, triangle_id, barycentrics);

    let light_distance = distance(ray_origin, light_hit.world_position);
    let ray_direction = (light_hit.world_position - ray_origin) / light_distance;
    let cos_theta_origin = saturate(dot(ray_direction, origin_world_normal));
    let cos_theta_light = saturate(dot(-ray_direction, light_hit.world_normal));
    let light_distance_squared = light_distance * light_distance;
    let irradiance = light_hit.material.emissive.rgb * cos_theta_origin * (cos_theta_light / light_distance_squared);

    let ray_t_max = light_distance - RAY_T_MIN;
    let ray = RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, RAY_NO_CULL, RAY_T_MIN, ray_t_max, ray_origin, ray_direction);
    var rq: ray_query;
    rayQueryInitialize(&rq, tlas, ray);
    rayQueryProceed(&rq);
    let ray_hit = rayQueryGetCommittedIntersection(&rq);
    let light_visible = f32(ray_hit.kind == RAY_QUERY_INTERSECTION_NONE);

    return irradiance * light_visible;
}

fn sample_light_sources(light_id: u32, light_count: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> LightSample {
    let light = light_sources[light_id];

    var sample: LightSample;
    if light.kind == LIGHT_SOURCE_DIRECTIONAL {
        sample = sample_directional_light(light.id, ray_origin, state);
    } else {
        sample = sample_emissive_triangle(light.id, light.kind, ray_origin, origin_world_normal, state);
    }

    sample.pdf /= f32(light_count);

    return sample;
}

fn trace_light_source(light_id: u32, ray_origin: vec3<f32>, origin_world_normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let light = light_sources[light_id];
    if light.kind == LIGHT_SOURCE_DIRECTIONAL {
        return trace_directional_light(light.id, ray_origin, state);
    } else {
        return trace_emissive_triangle(light.id, light.kind, ray_origin, origin_world_normal, state);
    }
}
