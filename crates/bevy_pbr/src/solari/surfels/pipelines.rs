use super::{view_resources::create_bind_group_layout, SurfelsSettings};
use crate::solari::{
    scene::SolariSceneBindGroupLayout,
    surfels::{SURFELS_SHADER_DESPAWN, SURFELS_SHADER_DIFFUSE, SURFELS_SHADER_SPAWN},
};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_render::render_resource::{
    BindGroupLayout, CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
    SpecializedComputePipeline, SpecializedComputePipelines,
};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum SurfelsKey {
    SpawnSurfels,
    SurfelsDiffuse,
    DebugSurfels,
    DespawnSurfels,
}

#[derive(Resource)]
pub struct SurfelsPipelines {
    pub scene_bind_group_layout: BindGroupLayout,
    pub view_bind_group_layout: BindGroupLayout,
}

impl FromWorld for SurfelsPipelines {
    fn from_world(world: &mut World) -> Self {
        let scene_bind_group_layout = world.resource::<SolariSceneBindGroupLayout>();
        let view_bind_group_layout = create_bind_group_layout(world.resource());

        Self {
            scene_bind_group_layout: scene_bind_group_layout.0.clone(),
            view_bind_group_layout,
        }
    }
}

impl SpecializedComputePipeline for SurfelsPipelines {
    type Key = SurfelsKey;

    fn specialize(&self, pass: Self::Key) -> ComputePipelineDescriptor {
        use SurfelsKey::*;

        let push_constant_ranges = vec![];
        let shader_defs = vec![];

        let (entry_point, shader) = match pass {
            SpawnSurfels => ("spawn_one_surfel", SURFELS_SHADER_SPAWN),
            SurfelsDiffuse => ("surfels_diffuse", SURFELS_SHADER_DIFFUSE),
            DebugSurfels => ("surfel_count", SURFELS_SHADER_DIFFUSE),
            DespawnSurfels => ("despawn_surfels", SURFELS_SHADER_DESPAWN),
        };

        ComputePipelineDescriptor {
            label: Some(format!("surfels_{entry_point}_pipeline").into()),
            layout: vec![
                self.scene_bind_group_layout.clone(),
                self.view_bind_group_layout.clone(),
            ],
            push_constant_ranges,
            shader,
            shader_defs,
            entry_point: entry_point.into(),
        }
    }
}

#[derive(Component)]
pub struct SurfelsPipelineIds {
    pub spawn_surfels: CachedComputePipelineId,
    pub surfels_diffuse: CachedComputePipelineId,
    pub debug_surfels: CachedComputePipelineId,
    pub despawn_surfels: CachedComputePipelineId,
}

pub fn prepare_pipelines(
    views: Query<
        Entity,
        (
            With<SurfelsSettings>,
            With<DepthPrepass>,
            With<NormalPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedComputePipelines<SurfelsPipelines>>,
    pipeline: Res<SurfelsPipelines>,
) {
    use SurfelsKey::*;

    let mut create_pipeline = |key| pipelines.specialize(&pipeline_cache, &pipeline, key);

    for entity in &views {
        commands.entity(entity).insert(SurfelsPipelineIds {
            spawn_surfels: create_pipeline(SpawnSurfels),
            surfels_diffuse: create_pipeline(SurfelsDiffuse),
            debug_surfels: create_pipeline(DebugSurfels),
            despawn_surfels: create_pipeline(DespawnSurfels),
        });
    }
}
