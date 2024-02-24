use super::{
    asset_binder::AssetBindings, scene_binder::SceneBindings, SolariSettings,
    SOLARI_SAMPLE_DIRECT_DIFFUSE_SHADER_HANDLE,
};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::QueryItem,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_render::{
    camera::ExtractedCamera,
    globals::{GlobalsBuffer, GlobalsUniform},
    render_graph::{NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{binding_types::*, *},
    renderer::{RenderContext, RenderDevice},
    texture::{CachedTexture, TextureCache},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};

pub struct SolariNode {
    bind_group_layout: BindGroupLayout,
    sample_direct_diffuse_pipeline: CachedComputePipelineId,
}

impl ViewNode for SolariNode {
    type ViewQuery = (
        &'static ViewResources,
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static ViewPrepassTextures,
        &'static ViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_resources, camera, view_target, view_prepass_textures, view_uniform_offset): QueryItem<
            Self::ViewQuery,
        >,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_bindings = world.resource::<AssetBindings>();
        let scene_bindings = world.resource::<SceneBindings>();
        let view_uniforms = world.resource::<ViewUniforms>();
        let globals_uniforms = world.resource::<GlobalsBuffer>();
        let (
            Some(sample_direct_diffuse_pipeline),
            Some(asset_bindings),
            Some(scene_bindings),
            Some(viewport),
            Some(gbuffer),
            Some(depth_buffer),
            Some(view_uniforms),
            Some(globals_uniforms),
        ) = (
            pipeline_cache.get_compute_pipeline(self.sample_direct_diffuse_pipeline),
            &asset_bindings.bind_group,
            &scene_bindings.bind_group,
            camera.physical_viewport_size,
            view_prepass_textures.deferred_view(),
            view_prepass_textures.depth_view(),
            view_uniforms.uniforms.binding(),
            globals_uniforms.buffer.binding(),
        )
        else {
            return Ok(());
        };

        let bind_group = render_context.render_device().create_bind_group(
            "solari_path_tracer_bind_group",
            &self.bind_group_layout,
            &BindGroupEntries::sequential((
                &view_resources.direct_diffuse.default_view,
                view_target.get_color_attachment().view,
                gbuffer,
                depth_buffer,
                view_uniforms,
                globals_uniforms,
            )),
        );

        let command_encoder = render_context.command_encoder();
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("solari"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, asset_bindings, &[]);
        pass.set_bind_group(1, scene_bindings, &[]);
        pass.set_bind_group(2, &bind_group, &[view_uniform_offset.offset]);

        pass.set_pipeline(sample_direct_diffuse_pipeline);
        pass.dispatch_workgroups((viewport.x + 7) / 8, (viewport.y + 7) / 8, 1);

        Ok(())
    }
}

impl FromWorld for SolariNode {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_bindings = world.resource::<AssetBindings>();
        let scene_bindings = world.resource::<SceneBindings>();

        let bind_group_layout = render_device.create_bind_group_layout(
            "solari_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                    texture_storage_2d(
                        ViewTarget::TEXTURE_FORMAT_HDR,
                        StorageTextureAccess::WriteOnly,
                    ),
                    texture_2d(TextureSampleType::Uint),
                    texture_depth_2d(),
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<GlobalsUniform>(false),
                ),
            ),
        );

        let sample_direct_diffuse_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("solari_sample_direct_diffuse_pipeline".into()),
                layout: vec![
                    asset_bindings.bind_group_layout.clone(),
                    scene_bindings.bind_group_layout.clone(),
                    bind_group_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: SOLARI_SAMPLE_DIRECT_DIFFUSE_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "sample_direct_diffuse".into(),
            });

        Self {
            bind_group_layout,
            sample_direct_diffuse_pipeline,
        }
    }
}

pub fn prepare_view_resources(
    query: Query<(Entity, &SolariSettings, &ExtractedCamera)>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, solari_settings, camera) in &query {
        if solari_settings.debug_path_tracer {
            continue;
        }
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };

        let direct_diffuse = TextureDescriptor {
            label: Some("solari_direct_diffuse_texture"),
            size: Extent3d {
                width: viewport.x,
                height: viewport.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };

        commands.entity(entity).insert(ViewResources {
            direct_diffuse: texture_cache.get(&render_device, direct_diffuse),
        });
    }
}

#[derive(Component)]
pub struct ViewResources {
    direct_diffuse: CachedTexture,
}
