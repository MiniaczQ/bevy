#import surfels::view_bindings allocated_surfel_ids_count, surfels_to_allocate, MAX_SURFELS, MAX_SPAWNS

@compute @workgroup_size(1)
fn approximate_spawns() {
    surfels_to_allocate.x = min(MAX_SURFELS - atomicLoad(&allocated_surfel_ids_count), MAX_SPAWNS);
    surfels_to_allocate.y = 1u;
    surfels_to_allocate.z = 1u;
}
