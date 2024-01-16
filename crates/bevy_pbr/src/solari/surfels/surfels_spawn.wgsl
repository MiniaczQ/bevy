#import surfels::view_bindings allocate_surfel, view, allocated_surfel_ids_count, surfel_position, surfel_normal, surfel_irradiance, INVALID_SURFEL_ID
#import bevy_solari::scene_bindings uniforms, map_ray_hit
#import surfels::utils trace_ray, rand_vec2f, rand_f

@compute @workgroup_size(1)
fn spawn_one_surfel() {
    var rng = uniforms.frame_count;
    let target_pos2d = rand_vec2f(&rng) * 2.0 - vec2<f32>(1.0);
    let target_pos = vec4<f32>(target_pos2d.x, target_pos2d.y, 0.1, 1.0);
    let world_target_pos = view.inverse_view_proj * target_pos;
    let direction = normalize(world_target_pos.xyz / world_target_pos.w - view.world_position);

    let ray_hit = trace_ray(view.world_position, direction, 0.1, 10000.0);
    if ray_hit.kind == RAY_QUERY_INTERSECTION_NONE {
        // Miss
        return;
    }

    let id = allocate_surfel();
    if id == INVALID_SURFEL_ID {
        // Out of surfels
        return;
    }

    let ray_hit_scene = map_ray_hit(ray_hit);
    surfel_position[id] = vec4<f32>(ray_hit_scene.world_position, 1.0);
    surfel_normal[id] = vec4<f32>(ray_hit_scene.world_normal, 1.0);
    surfel_irradiance[id] = vec4<f32>(0.0);
}
