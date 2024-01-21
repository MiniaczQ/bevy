#import surfels::view_bindings view, unallocated_surfel_ids_stack, allocated_surfels_bitmap, unallocated_surfels, surfel_position, surfel_normal, surfel_irradiance, depth_buffer, normals_buffer, MAX_SPAWNS, MAX_SURFELS, SurfelIrradiance
#import bevy_solari::scene_bindings uniforms
#import surfels::utils rand_vec2f, depth_to_world_position
#import bevy_pbr::utils hsv2rgb

fn allocate_surfel() -> u32 {
    let idx = atomicSub(&unallocated_surfels, 1u) - 1u;
    let id = unallocated_surfel_ids_stack[idx];

    let bin = id / 32u;
    let bit = id % 32u;
    atomicOr(&allocated_surfels_bitmap[bin], (1u << bit));
    return id;
}

@compute @workgroup_size(1)
fn spawn_one_surfel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var rng = uniforms.frame_count * MAX_SPAWNS + global_id.x;
    let screen_uv = rand_vec2f(&rng);
    let pixel_uv = vec2<u32>(screen_uv * view.viewport.zw);
    let clip_z = textureLoad(depth_buffer, pixel_uv, 0i);
    if(clip_z == 0.0) { return; } // Miss

    let world_xyz = depth_to_world_position(clip_z, screen_uv);
    let id = allocate_surfel();
    surfel_position[id] = vec4<f32>(world_xyz, 1.0);
    let normal = normalize(textureLoad(normals_buffer, pixel_uv, 0i).xyz * 2.0 - 1.0);
    surfel_normal[id] = vec4<f32>(normal, 1.0);
    let color = hsv2rgb(f32(id) / f32(MAX_SURFELS), 1.0, 0.5);
    surfel_irradiance[id] = SurfelIrradiance(color, vec3<f32>(0.0), 0u);
}



//    old method that uses ray query

//    let clip_xyzw = vec4<f32>(clip_xy, 0.1, 1.0);
//    let world_xyzw = view.inverse_view_proj * clip_xyzw;
//    let direction = normalize(world_xyzw.xyz / world_xyzw.w - view.world_position);
//    let ray_hit = trace_ray(view.world_position, direction, 0.1, 10000.0);
//    if ray_hit.kind == RAY_QUERY_INTERSECTION_NONE {
//        // Miss
//        return;
//    }
//    let ray_hit_scene = map_ray_hit(ray_hit);
//    surfel_position[id] = vec4<f32>(ray_hit_scene.world_position, 1.0);
//    surfel_normal[id] = vec4<f32>(ray_hit_scene.world_normal, 1.0);
//    surfel_irradiance[id] = vec4<f32>(0.0);
