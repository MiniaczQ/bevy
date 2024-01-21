#import surfels::view_bindings allocated_surfels_bitmap, surfel_position, surfel_normal, surfel_irradiance, MAX_SURFELS
#import bevy_solari::scene_bindings map_ray_hit, uniforms
#import surfels::utils sample_cosine_hemisphere, sample_direct_lighting, trace_ray

@compute @workgroup_size(64)
fn surfels_update(@builtin(local_invocation_index) local_idx: u32) {
    var id = local_idx * 32u;
    let max_id = id + MAX_SURFELS / 32u;
    for (; id < max_id; id++) {
        if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

        let surfel_pos = surfel_position[id].xyz;
        let surfel_nor = surfel_normal[id].xyz;
        var surfel_irr = surfel_irradiance[id];
        var rng = uniforms.frame_count * MAX_SURFELS + local_idx;
        var irradiance = vec3<f32>(0.0);

        irradiance += sample_direct_lighting(surfel_pos, surfel_nor, &rng);

        let ray_dir = sample_cosine_hemisphere(surfel_nor, &rng);
        let ray_hit = trace_ray(surfel_pos, ray_dir, 0.001, 100.0);
        if ray_hit.kind != RAY_QUERY_INTERSECTION_NONE {
            let hit_dis = 0.001 + ray_hit.t * 100.0;
            let hit_pos = surfel_pos + ray_dir * hit_dis;
            let solari_ray_hit = map_ray_hit(ray_hit);

            var lighting = vec3<f32>(0.0);
            var total_weight = 0.000001;
            for (var id = 0u; id < MAX_SURFELS; id++) {
                if (allocated_surfels_bitmap[id / 32u] & (1u << (id % 32u))) == 0u { continue; } // Surfel not active

                let surfel_pos2 = surfel_position[id].xyz;
                let surfel_dis2 = distance(hit_pos, surfel_pos2);
                let weight = max(1.0 - surfel_dis2, 0.0);
                lighting += surfel_irradiance[id].mean * weight;
                total_weight += weight;
            }

            let surfel_illu = lighting / total_weight;
            let attenuation = (1.0 + hit_dis) * (1.0 + hit_dis);
            irradiance += solari_ray_hit.material.base_color * surfel_illu / attenuation;
        }

        // Welford's online algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        surfel_irr.probes = min(surfel_irr.probes + 1u, 10u);
        let delta = irradiance - surfel_irr.mean;
        surfel_irr.mean += delta / f32(surfel_irr.probes);
        let delta2 = irradiance - surfel_irr.mean;
        surfel_irr.mean_squared += delta * delta2;
        
        surfel_irradiance[id] = surfel_irr;
    }
}
