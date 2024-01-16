pub use self::node::SurfelsNode;
use self::{
    pipelines::{prepare_pipelines, SurfelsPipelines},
    view_resources::{prepare_bind_groups, prepare_resources},
};
use super::SolariEnabled;
use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::CORE_3D;
use bevy_ecs::{component::Component, prelude::resource_exists, schedule::IntoSystemConfigs};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::{Shader, SpecializedComputePipelines},
    Render, RenderApp, RenderSet,
};
pub(crate) use view_resources::SurfelsViewResources;

mod node;
mod pipelines;
mod view_resources;

const MAX_SURFELS: u64 = 1024;

const SURFELS_SHADER_VIEW_BINDINGS: Handle<Shader> = Handle::weak_from_u128(1_531_537_373_000);
const SURFELS_SHADER_SPAWN: Handle<Shader> = Handle::weak_from_u128(1_531_537_373_001);
const SURFELS_SHADER_DESPAWN: Handle<Shader> = Handle::weak_from_u128(1_531_537_373_002);
const SURFELS_SHADER_PBR: Handle<Shader> = Handle::weak_from_u128(1_531_537_373_003);
const SURFELS_SHADER_UTILS: Handle<Shader> = Handle::weak_from_u128(1_531_537_373_004);

pub struct SurfelsPlugin;

impl Plugin for SurfelsPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SURFELS_SHADER_VIEW_BINDINGS,
            "view_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SURFELS_SHADER_SPAWN,
            "surfels_spawn.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SURFELS_SHADER_DESPAWN,
            "surfels_despawn.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SURFELS_SHADER_PBR,
            "surfels_pbr.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(app, SURFELS_SHADER_UTILS, "utils.wgsl", Shader::from_wgsl);

        app.add_plugins(ExtractComponentPlugin::<SurfelsSettings>::default());

        app.sub_app_mut(RenderApp)
            .add_render_graph_node::<ViewNodeRunner<SurfelsNode>>(CORE_3D, "surfels")
            .add_render_graph_edges(
                CORE_3D,
                &[
                    // PREPASS -> SURFELS -> MAIN_PASS
                    bevy_core_pipeline::core_3d::graph::node::PREPASS,
                    "surfels",
                    bevy_core_pipeline::core_3d::graph::node::START_MAIN_PASS,
                ],
            )
            .init_resource::<SurfelsPipelines>()
            .init_resource::<SpecializedComputePipelines<SurfelsPipelines>>()
            .add_systems(
                Render,
                (
                    prepare_pipelines.in_set(RenderSet::PrepareResources),
                    prepare_resources.in_set(RenderSet::PrepareResources),
                    prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
                )
                    .run_if(resource_exists::<SolariEnabled>()),
            );
    }
}

#[derive(Component, ExtractComponent, Clone, Default)]
pub struct SurfelsSettings {}
