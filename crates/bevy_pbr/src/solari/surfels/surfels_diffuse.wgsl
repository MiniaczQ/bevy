#import surfels::view_bindings view, depth_buffer, diffuse_irradiance_output, allocated_surfels_bitmap, surfel_position, surfel_irradiance, MAX_SURFELS
#import surfels::utils depth_to_world_position
#import bevy_pbr::utils hsv2rgb

const SCALE: f32 = 0.0078125;
const AFFECTION_RANGE: f32 = 0.1;

@compute @workgroup_size(8, 8)
fn surfels_debug_diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2<u32>(view.viewport.zw)) { return; } // Outside of view

    let clip_z = textureLoad(depth_buffer, global_id.xy, 0i);
    let screen_xy = (vec2<f32>(global_id.xy) + 0.5) / view.viewport.zw;
    let world_pos = depth_to_world_position(clip_z, screen_xy);

    let view_distance = distance(view.world_position, world_pos);
    let radius = view_distance * SCALE;
    
    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active
        let surfel_pos = surfel_position[id].xyz;
        if distance(world_pos, surfel_pos) < radius {
            let color = hsv2rgb(f32(id) / f32(MAX_SURFELS), 1.0, 0.5);
            textureStore(diffuse_irradiance_output, global_id.xy, vec4<f32>(color, 1.0));
            return;
        }
    }

    textureStore(diffuse_irradiance_output, global_id.xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn surfels_diffuse(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id.xy >= vec2<u32>(view.viewport.zw)) { return; } // Outside of view

    let screen_z = textureLoad(depth_buffer, global_id.xy, 0i);
    if screen_z == 0.0 { return; } // Miss

    let screen_xy = (vec2<f32>(global_id.xy) + 0.5) / view.viewport.zw;
    let world_pos = depth_to_world_position(screen_z, screen_xy);
    let view_dis = distance(view.world_position, world_pos);
    let affection_dis = view_dis * AFFECTION_RANGE;
    
    var contributors = 0;
    var lighting = vec3<f32>(0.0);

    for (var id = 0u; id < MAX_SURFELS; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

        let surfel_pos = surfel_position[id].xyz;
        let surfel_dis = distance(world_pos, surfel_pos);
        if surfel_dis < affection_dis {
            contributors += 1;
            lighting += surfel_irradiance[id].mean;
        }
    }

    if contributors > 0 { lighting = lighting / f32(contributors); }
    textureStore(diffuse_irradiance_output, global_id.xy, vec4<f32>(lighting, 1.0));
}

@compute @workgroup_size(1)
fn surfel_count() {
    let sx = 16u;
    let sy = MAX_SURFELS / sx;
    let size = 5u;
    for(var x = 0u; x < sx; x++) {
        for(var y = 0u; y < sy; y++) {
            let screen_xy = vec2<u32>(x, y);
            let id = x * sy + y;
            var color = vec4<f32>(0.0);
            if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) != 0u {
                color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }
            for(var ox = 0u; ox < size; ox++) {
                for(var oy = 0u; oy < size; oy++) {
                    textureStore(diffuse_irradiance_output, screen_xy * size + vec2<u32>(ox, oy), color);
                }
            }
        }
    }
}
