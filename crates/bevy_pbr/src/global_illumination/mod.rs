mod asset_binder;
mod blas_manager;
mod extract_asset_events;
mod gpu_types;
mod scene_binder;
pub mod surfels;

use self::{
    asset_binder::{prepare_asset_binding_arrays, AssetBindings},
    blas_manager::{prepare_new_blas, BlasManager},
    extract_asset_events::{
        extract_asset_events, ExtractAssetEventsSystemState, ExtractedAssetEvents,
    },
    graph::NodeGi,
    scene_binder::{extract_scene, prepare_scene_bindings, ExtractedScene, SceneBindings},
    surfels::{prepare_view_resources, GlobalIlluminationNode},
};
use crate::{graph::NodePbr, DefaultOpaqueRendererMethod};
use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{
    component::Component,
    schedule::{common_conditions::any_with_component, IntoSystemConfigs},
    system::Resource,
};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    mesh::Mesh,
    render_asset::prepare_assets,
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::Shader,
    renderer::RenderDevice,
    settings::WgpuFeatures,
    texture::Image,
    view::Msaa,
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_utils::tracing::warn;

pub mod graph {
    use bevy_render::render_graph::RenderLabel;

    #[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
    pub enum NodeGi {
        Surfels,
    }
}

const MAX_SURFELS: u64 = 1024;
const BINDINGS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(1717171717171717);
const SURFELS_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(1_531_537_373_001);

/// TODO: Docs
pub struct GlobalIlluminationPlugin;

impl Plugin for GlobalIlluminationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Msaa::Off)
            .insert_resource(DefaultOpaqueRendererMethod::deferred());

        load_internal_asset!(
            app,
            BINDINGS_SHADER_HANDLE,
            "bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SURFELS_SHADER_HANDLE,
            "surfels.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        match app.world.get_resource::<RenderDevice>() {
            Some(render_device) => {
                if !render_device.features().contains(Self::required_features()) {
                    let missing = Self::required_features().difference(render_device.features());
                    warn!(?missing, "Missing features");
                    return;
                }
            }
            _ => {}
        }

        app.insert_resource(GlobalIlluminationSupported)
            .init_resource::<ExtractAssetEventsSystemState>()
            .add_plugins(ExtractComponentPlugin::<GlobalIlluminationSettings>::default());

        let render_app = app.get_sub_app_mut(RenderApp).unwrap();
        render_app
            .init_resource::<ExtractedAssetEvents>()
            .init_resource::<ExtractedScene>()
            .init_resource::<BlasManager>()
            .init_resource::<AssetBindings>()
            .init_resource::<SceneBindings>()
            .add_systems(ExtractSchedule, (extract_asset_events, extract_scene))
            .add_systems(
                Render,
                (
                    prepare_new_blas
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Mesh>),
                    prepare_asset_binding_arrays
                        .in_set(RenderSet::PrepareAssets)
                        .after(prepare_assets::<Mesh>)
                        .after(prepare_assets::<Image>),
                    prepare_view_resources.in_set(RenderSet::PrepareResources),
                    prepare_scene_bindings.in_set(RenderSet::PrepareBindGroups),
                )
                    .run_if(any_with_component::<GlobalIlluminationSettings>),
            )
            .add_render_graph_node::<ViewNodeRunner<GlobalIlluminationNode>>(
                Core3d,
                NodeGi::Surfels,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPass,
                    NodeGi::Surfels,
                    NodePbr::DeferredLightingPass,
                ),
            );
    }
}

impl GlobalIlluminationPlugin {
    /// TODO: Docs
    pub fn required_features() -> WgpuFeatures {
        WgpuFeatures::RAY_TRACING_ACCELERATION_STRUCTURE
            | WgpuFeatures::RAY_QUERY
            | WgpuFeatures::TEXTURE_BINDING_ARRAY
            | WgpuFeatures::BUFFER_BINDING_ARRAY
            | WgpuFeatures::STORAGE_RESOURCE_BINDING_ARRAY
            | WgpuFeatures::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | WgpuFeatures::PARTIALLY_BOUND_BINDING_ARRAY
            | WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | WgpuFeatures::PUSH_CONSTANTS
    }
}

/// TODO: Docs
#[derive(Resource)]
pub struct GlobalIlluminationSupported;

/// TODO: Docs
// Requires MSAA off, HDR, CameraMainTextureUsages::with_storage_binding(), deferred + depth + motion vector prepass,
//   DefaultOpaqueRendererMethod::deferred, and should disable shadows for all lights
#[derive(Component, ExtractComponent, Clone)]
pub struct GlobalIlluminationSettings;

impl Default for GlobalIlluminationSettings {
    fn default() -> Self {
        Self
    }
}
