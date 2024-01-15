#define_import_path surfels::pbr

#import bevy_pbr::mesh_view_bindings view, allocated_surfel_ids_stack, allocated_surfels_count, surfel_position, surfel_irradiance, SURFEL_STACK_SIZE
#import bevy_pbr::utils hsv2rgb

const SCALE: f32 = 0.01;

fn pbr(output_color: vec4<f32>, world_position: vec4<f32>, frag_coord: vec4<f32>) -> vec4<f32> {
    let pos = world_position.xyz;
    let view_distance = distance(view.world_position, pos);
    let radius = view_distance * SCALE;
    let surfel_count = atomicLoad(&allocated_surfels_count);

    //let out = debug_line(output_color, frag_coord, 10.0, f32(surfel_count));

    for (var i = 0u; i < surfel_count; i++) {
        let id = allocated_surfel_ids_stack[i];
        let surfel_pos = surfel_position[id];
        if distance(pos, surfel_pos) < radius {
            let color = hsv2rgb(f32(id) / f32(SURFEL_STACK_SIZE), 1.0, 0.5);
            return vec4<f32>(color, 1.0);
        }
    }

    return output_color;
}

fn debug_line(output_color: vec4<f32>, frag_coord: vec4<f32>, idx: f32, value: f32) -> vec4<f32> {
    if (frag_coord.y + 1.0) < frag_coord.y || frag_coord.y < (idx - 1.0)  {
        return output_color;
    }
    if 10.0 < frag_coord.x && frag_coord.x <= 11.0 + f32(value) {
        return vec4<f32>(1.0);
    }
    return output_color;
}
