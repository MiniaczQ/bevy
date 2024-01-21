#import surfels::view_bindings view, unallocated_surfel_ids_stack, allocated_surfel_ids_count, allocated_surfels_bitmap, surfel_position, MAX_SURFELS

fn deallocate_surfel(id: u32) {
    let idx = atomicSub(&allocated_surfel_ids_count, 1u) - 1u; // value post operation
    unallocated_surfel_ids_stack[idx] = id;

    let bin = id / 32u;
    let bit = id % 32u;
    allocated_surfels_bitmap[bin] = allocated_surfels_bitmap[bin] & ~(1u << bit);
}

@compute @workgroup_size(32)
fn despawn_surfels(@builtin(local_invocation_index) local_idx: u32) {
    var id = local_idx * 32u;
    let max_id = id + MAX_SURFELS / 32u;
    for (; id < max_id; id++) {
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
