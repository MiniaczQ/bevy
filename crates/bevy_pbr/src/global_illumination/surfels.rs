use std::num::NonZeroU64;

use super::{
    asset_binder::AssetBindings, scene_binder::SceneBindings, GlobalIlluminationSettings,
    MAX_SURFELS, SURFELS_SHADER_HANDLE,
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
    view::{ViewUniform, ViewUniformOffset, ViewUniforms},
};
use buffer_cache::{BufferCache, CachedBuffer};

pub struct GlobalIlluminationNode {
    bind_group_layout_atomic_bitmap: BindGroupLayout,
    bind_group_layout: BindGroupLayout,
    count_unallocated_surfels: CachedComputePipelineId,
    spawn_surfels: CachedComputePipelineId,
    update_surfels: CachedComputePipelineId,
    apply_surfel_diffuse: CachedComputePipelineId,
    debug_surfels_view: CachedComputePipelineId,
    despawn_surfels: CachedComputePipelineId,
}

impl ViewNode for GlobalIlluminationNode {
    type ViewQuery = (
        &'static GlobalIlluminationViewResources,
        &'static ExtractedCamera,
        &'static ViewPrepassTextures,
        &'static ViewUniformOffset,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (view_resources, camera, view_prepass_textures, view_uniform_offset): QueryItem<
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
            Some(count_unallocated_surfels),
            Some(spawn_surfels),
            Some(update_surfels),
            Some(apply_surfel_diffuse),
            Some(debug_surfels_view),
            Some(despawn_surfels),
            Some(asset_bind_group),
            Some(scene_bind_group),
            Some(viewport),
            Some(gbuffer),
            Some(depth_buffer),
            Some(view_uniforms),
            Some(globals_uniforms),
        ) = (
            pipeline_cache.get_compute_pipeline(self.count_unallocated_surfels),
            pipeline_cache.get_compute_pipeline(self.spawn_surfels),
            pipeline_cache.get_compute_pipeline(self.update_surfels),
            pipeline_cache.get_compute_pipeline(self.apply_surfel_diffuse),
            pipeline_cache.get_compute_pipeline(self.debug_surfels_view),
            pipeline_cache.get_compute_pipeline(self.despawn_surfels),
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

        let bind_group_indirect_allocation = render_context.render_device().create_bind_group(
            "surfels_bind_group_layout_indirect_allocation",
            &self.bind_group_layout_atomic_bitmap,
            &BindGroupEntries::sequential((
                view_uniforms.clone(),
                globals_uniforms.clone(),
                depth_buffer,
                gbuffer,
                &view_resources.unallocated_surfel_ids_stack,
                &view_resources.allocated_surfels_bitmap,
                &view_resources.unallocated_surfels,
                &view_resources.surfels_surface,
                &view_resources.surfels_irradiance,
                &view_resources.diffuse_output.default_view,
                &view_resources.surfels_to_allocate,
            )),
        );

        let bind_group = render_context.render_device().create_bind_group(
            "surfels_bind_group",
            &self.bind_group_layout,
            &BindGroupEntries::sequential((
                view_uniforms,
                globals_uniforms,
                depth_buffer,
                gbuffer,
                &view_resources.unallocated_surfel_ids_stack,
                &view_resources.allocated_surfels_bitmap,
                &view_resources.unallocated_surfels,
                &view_resources.surfels_surface,
                &view_resources.surfels_irradiance,
                &view_resources.diffuse_output.default_view,
            )),
        );

        let command_encoder = render_context.command_encoder();
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("surfels"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, asset_bind_group, &[]);
        pass.set_bind_group(1, scene_bind_group, &[]);
        pass.set_bind_group(
            2,
            &bind_group_indirect_allocation,
            &[view_uniform_offset.offset],
        );

        pass.push_debug_group("count_unallocated_surfels");
        pass.set_pipeline(count_unallocated_surfels);
        pass.dispatch_workgroups(1, 1, 1);
        pass.pop_debug_group();

        pass.set_bind_group(2, &bind_group, &[view_uniform_offset.offset]);

        pass.push_debug_group("spawn_surfels");
        pass.set_pipeline(spawn_surfels);
        pass.dispatch_workgroups_indirect(&view_resources.surfels_to_allocate.buffer, 0);
        pass.pop_debug_group();

        pass.push_debug_group("surfels_update");
        pass.set_pipeline(update_surfels);
        pass.dispatch_workgroups(MAX_SURFELS as u32 / 32, 1, 1);
        pass.pop_debug_group();

        pass.push_debug_group("apply_surfel_diffuse");
        pass.set_pipeline(apply_surfel_diffuse);
        pass.dispatch_workgroups((viewport.x + 7) / 8, (viewport.y + 7) / 8, 1);
        pass.pop_debug_group();

        //pass.push_debug_group("debug_surfels_view");
        //pass.set_pipeline(debug_surfels_view);
        //pass.dispatch_workgroups((viewport.x + 7) / 8, (viewport.y + 7) / 8, 1);
        //pass.pop_debug_group();

        pass.push_debug_group("despawn_surfels");
        pass.set_pipeline(despawn_surfels);
        pass.dispatch_workgroups(1, 1, 1);
        pass.pop_debug_group();

        Ok(())
    }
}

impl FromWorld for GlobalIlluminationNode {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let asset_bindings = world.resource::<AssetBindings>();
        let scene_bindings = world.resource::<SceneBindings>();

        let bind_group_layout_indirect_allocation = render_device.create_bind_group_layout(
            "surfels_bind_group_layout_indirect_allocation",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<GlobalsUniform>(false),
                    texture_depth_2d(),                  // depth
                    texture_2d(TextureSampleType::Uint), // gbuffer
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(4 * MAX_SURFELS) }),
                    ), // stack
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(4 * MAX_SURFELS / 32) }),
                    ), // bitmap
                    storage_buffer_sized(false, Some(unsafe { NonZeroU64::new_unchecked(4) })), // stack pointer
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(48 * MAX_SURFELS) }),
                    ), // surface
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(32 * MAX_SURFELS) }),
                    ), // irradiance
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite), // output
                    storage_buffer_sized(false, Some(unsafe { NonZeroU64::new_unchecked(12) })), // surfels to allocate
                ),
            ),
        );

        let bind_group_layout = render_device.create_bind_group_layout(
            "surfels_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<ViewUniform>(true),
                    uniform_buffer::<GlobalsUniform>(false),
                    texture_depth_2d(),                  // depth
                    texture_2d(TextureSampleType::Uint), // gbuffer
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(4 * MAX_SURFELS) }),
                    ), // stack
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(4 * MAX_SURFELS / 32) }),
                    ), // bitmap
                    storage_buffer_sized(false, Some(unsafe { NonZeroU64::new_unchecked(4) })), // stack pointer
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(48 * MAX_SURFELS) }),
                    ), // position
                    storage_buffer_sized(
                        false,
                        Some(unsafe { NonZeroU64::new_unchecked(32 * MAX_SURFELS) }),
                    ), // irradiance
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite), // output
                ),
            ),
        );

        let count_unallocated_surfels =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("count_unallocated_surfels_pipeline".into()),
                layout: vec![
                    asset_bindings.bind_group_layout.clone(),
                    scene_bindings.bind_group_layout.clone(),
                    bind_group_layout_indirect_allocation.clone(),
                ],
                push_constant_ranges: vec![],
                shader: SURFELS_SHADER_HANDLE,
                shader_defs: vec!["INDIRECT_ALLOCATE".into()],
                entry_point: "count_unallocated_surfels".into(),
            });

        let spawn_surfels = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("spawn_surfels_pipeline".into()),
            layout: vec![
                asset_bindings.bind_group_layout.clone(),
                scene_bindings.bind_group_layout.clone(),
                bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SURFELS_SHADER_HANDLE,
            shader_defs: vec!["ATOMIC_BITMAP".into()],
            entry_point: "spawn_surfels".into(),
        });

        let update_surfels = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("update_surfels_pipeline".into()),
            layout: vec![
                asset_bindings.bind_group_layout.clone(),
                scene_bindings.bind_group_layout.clone(),
                bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SURFELS_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "update_surfels".into(),
        });

        let apply_surfel_diffuse =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("apply_surfel_diffuse_pipeline".into()),
                layout: vec![
                    asset_bindings.bind_group_layout.clone(),
                    scene_bindings.bind_group_layout.clone(),
                    bind_group_layout.clone(),
                ],
                push_constant_ranges: vec![],
                shader: SURFELS_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "apply_surfel_diffuse".into(),
            });

        let debug_surfels_view = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("debug_surfels_view_pipeline".into()),
            layout: vec![
                asset_bindings.bind_group_layout.clone(),
                scene_bindings.bind_group_layout.clone(),
                bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SURFELS_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "debug_surfels_view".into(),
        });

        let despawn_surfels = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("despawn_surfels_pipeline".into()),
            layout: vec![
                asset_bindings.bind_group_layout.clone(),
                scene_bindings.bind_group_layout.clone(),
                bind_group_layout.clone(),
            ],
            push_constant_ranges: vec![],
            shader: SURFELS_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "despawn_surfels".into(),
        });

        Self {
            bind_group_layout_atomic_bitmap: bind_group_layout_indirect_allocation,
            bind_group_layout,
            count_unallocated_surfels,
            spawn_surfels,
            update_surfels,
            apply_surfel_diffuse,
            debug_surfels_view,
            despawn_surfels,
        }
    }
}

pub fn prepare_view_resources(
    query: Query<(Entity, &GlobalIlluminationSettings, &ExtractedCamera)>,
    mut texture_cache: ResMut<TextureCache>,
    mut buffer_cache: ResMut<BufferCache>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, _solari_settings, camera) in &query {
        let Some(viewport) = camera.physical_viewport_size else {
            continue;
        };

        let unallocated_surfel_ids_stack = BufferDescriptor {
            label: Some("unallocated_surfel_ids_stack"),
            size: 4 * MAX_SURFELS,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let allocated_surfels_bitmap = BufferDescriptor {
            label: Some("allocated_surfels_bitmap"),
            size: 4 * MAX_SURFELS / 32,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let unallocated_surfels = BufferDescriptor {
            label: Some("unallocated_surfels"),
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let surfels_surface = BufferDescriptor {
            label: Some("surfels_surface"),
            size: 48 * MAX_SURFELS,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let surfels_irradiance = BufferDescriptor {
            label: Some("surfels_irradiance"),
            size: 32 * MAX_SURFELS,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        };
        let diffuse_output = TextureDescriptor {
            label: Some("global_illumination_diffuse_output"),
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
        let surfels_to_allocate = BufferDescriptor {
            label: Some("surfels_to_allocate"),
            size: 12,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        };

        fn init_stack() -> Vec<u8> {
            (0u32..MAX_SURFELS as u32)
                .flat_map(|v| v.to_le_bytes())
                .collect()
        }

        fn init_stack_ptr() -> Vec<u8> {
            (MAX_SURFELS as u32).to_le_bytes().to_vec()
        }

        let unallocated_surfel_ids_stack =
            buffer_cache.get_or(&render_device, unallocated_surfel_ids_stack, init_stack);
        let allocated_surfels_bitmap = buffer_cache.get(&render_device, allocated_surfels_bitmap);
        let unallocated_surfels =
            buffer_cache.get_or(&render_device, unallocated_surfels, init_stack_ptr);
        let surfels_surface = buffer_cache.get(&render_device, surfels_surface);
        let surfel_irradiance = buffer_cache.get(&render_device, surfels_irradiance);
        let diffuse_output = texture_cache.get(&render_device, diffuse_output);
        let surfels_to_allocate = buffer_cache.get(&render_device, surfels_to_allocate);

        commands
            .entity(entity)
            .insert(GlobalIlluminationViewResources {
                unallocated_surfel_ids_stack,
                allocated_surfels_bitmap,
                unallocated_surfels,
                surfels_surface,
                surfels_irradiance: surfel_irradiance,
                diffuse_output,
                surfels_to_allocate,
            });
    }
}

#[derive(Component)]
pub struct GlobalIlluminationViewResources {
    pub unallocated_surfel_ids_stack: CachedBuffer,
    pub allocated_surfels_bitmap: CachedBuffer,
    pub unallocated_surfels: CachedBuffer,
    pub surfels_surface: CachedBuffer,
    pub surfels_irradiance: CachedBuffer,
    pub surfels_to_allocate: CachedBuffer,
    pub diffuse_output: CachedTexture,
}
