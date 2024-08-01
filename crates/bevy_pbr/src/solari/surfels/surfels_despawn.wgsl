#import surfels::view_bindings view, unallocated_surfel_ids_stack, unallocated_surfels, allocated_surfels_bitmap, surfel_position, MAX_SURFELS

fn deallocate_surfel(id: u32) {
    let idx = atomicAdd(&unallocated_surfels, 1u);
    unallocated_surfel_ids_stack[idx] = id;

    let bin = id / 32u;
    let bit = id % 32u;
    allocated_surfels_bitmap[bin] = allocated_surfels_bitmap[bin] & ~(1u << bit);
}

// Frustum despawn
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

// Importance despawn


// Density despawn
// @compute @workgroup_size(8, 8)
// fn density_despawn_surfels(@builtin(local_invocation_id) local_id: vec3<u32>) {
//     let start = 0.125 * f32(local_id.xy);
//     let end = start + 0.125;
//     let expected_density = f32(MAX_SURFELS) / 64.0;
//     var count = 0;
//     for (var id = 0; id < MAX_SURFELS; id++) {
//         if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u {
//             // Surfel not active
//             continue;
//         }
//         let world_surfel_pos = surfel_position[id];
//         let ss_surfel_pos = view.view_proj * world_surfel_pos;
//         let pos = ss_surfel_pos.xyz / ss_surfel_pos.w;
//         // Check rectangle
//         if pos.x < start.x || end.x < pos.x || pos.y < start.y || end.y < pos.y {
//             continue;
//         }
// 
//         // Replace with reservoir
//         if (f32(count) > expected_density) {
//             deallocate_surfel(id);
//         }
// 
//         count = count + 1;
//     }
// }
