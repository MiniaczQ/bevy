#import surfels::view_bindings deallocate_surfel, view, allocated_surfels_bitmap, surfel_position, MAX_SURFELS

@compute @workgroup_size(1)
fn despawn_surfels() {
    for (var id = 0u; id < MAX_SURFELS; id++) {
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
