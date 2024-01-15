use super::{pipelines::SurfelsPipelineIds, view_resources::SurfelsBindGroups};
use crate::solari::scene::SolariSceneBindGroup;
use bevy_ecs::{query::QueryItem, world::World};
use bevy_render::{
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
    view::ViewUniformOffset,
};

#[derive(Default)]
pub struct SurfelsNode;

impl ViewNode for SurfelsNode {
    type ViewQuery = (
        &'static SurfelsPipelineIds,
        &'static SurfelsBindGroups,
        &'static ViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (pipeline_ids, bind_groups, view_uniform_offset): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipeline_cache), Some(SolariSceneBindGroup(Some(scene_bind_group)))) = (
            world.get_resource::<PipelineCache>(),
            world.get_resource::<SolariSceneBindGroup>(),
        ) else {
            return Ok(());
        };
        let (Some(spawn_surfels_pipeline), Some(despawn_surfels_pipeline)) = (
            pipeline_cache.get_compute_pipeline(pipeline_ids.spawn_surfels),
            pipeline_cache.get_compute_pipeline(pipeline_ids.despawn_surfels),
        ) else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();

        let mut surfels_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surfels_pass"),
            timestamp_writes: None,
        });

        surfels_pass.set_bind_group(0, scene_bind_group, &[]);
        surfels_pass.set_bind_group(
            1,
            &bind_groups.view_bind_group,
            &[view_uniform_offset.offset],
        );

        surfels_pass.push_debug_group("spawn_surfels");
        surfels_pass.set_pipeline(spawn_surfels_pipeline);
        surfels_pass.dispatch_workgroups(1, 1, 1);
        surfels_pass.pop_debug_group();

        surfels_pass.push_debug_group("despawn_surfels");
        surfels_pass.set_pipeline(despawn_surfels_pipeline);
        surfels_pass.dispatch_workgroups(1, 1, 1);
        surfels_pass.pop_debug_group();

        Ok(())
    }
}
