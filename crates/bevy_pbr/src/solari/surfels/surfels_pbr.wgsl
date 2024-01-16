#define_import_path surfels::pbr

#import bevy_pbr::mesh_view_bindings view, allocated_surfels_bitmap, allocated_surfel_ids_count, surfel_position, surfel_irradiance, MAX_SURFELS
#import bevy_pbr::utils hsv2rgb

const SCALE: f32 = 0.0078125;

fn pbr(output_color: vec4<f32>, world_position: vec4<f32>, frag_coord: vec4<f32>) -> vec4<f32> {
    let pos = world_position.xyz;
    let view_distance = distance(view.world_position, pos);
    let radius = view_distance * SCALE;
    let surfel_count = atomicLoad(&allocated_surfel_ids_count);
    var out = output_color;

    //out = debug_line(out, frag_coord, 10.0, 5.0);
    
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u {
            // Surfel not active
            continue;
        }
        let surfel_pos = surfel_position[id].xyz;
        if distance(pos, surfel_pos) < radius {
            let color = hsv2rgb(f32(id) / f32(MAX_SURFELS), 1.0, 0.5);
            return vec4<f32>(color, 1.0);
        }
    }

    return out;
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

//const DBG_SIZE: f32 = 1.0;
//
//fn debug_bitmap(output_color: vec4<f32>, frag_coord: vec4<f32>, offset: vec2<f32>, value: f32) -> vec4<f32> {
//    for (var i = 0; i < 32; i++) {
//        var mask = 1u << i;
//        var v = value & mask;
//    }
//    return output_color;
//}
