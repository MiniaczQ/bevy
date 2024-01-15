use super::{SurfelsPipelines, SurfelsSettings, SURFEL_STACK_SIZE};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferCache,
        BufferDescriptor, BufferUsages, CachedBuffer, ShaderStages, ShaderType,
    },
    renderer::{RenderDevice, RenderQueue},
    view::{ViewUniform, ViewUniforms},
};
use std::num::NonZeroU64;

#[derive(Component)]
pub struct SurfelsViewResources {
    unallocated_surfel_ids_stack: CachedBuffer,
    pub allocated_surfel_ids_stack: CachedBuffer,
    pub allocated_surfels_count: CachedBuffer,
    pub surfel_position: CachedBuffer,
    surfel_normal: CachedBuffer,
    pub surfel_irradiance: CachedBuffer,
}

pub fn prepare_resources(
    views: Query<
        Entity,
        (
            With<ExtractedCamera>,
            With<SurfelsSettings>,
            With<DepthPrepass>,
            With<NormalPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    mut commands: Commands,
    mut buffer_cache: ResMut<BufferCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let buffer = |label, size| BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    };

    for entity in &views {
        let mut unallocated_surfel_ids_stack =
            buffer("unallocated_surfel_ids_stack", 4 * SURFEL_STACK_SIZE);
        unallocated_surfel_ids_stack.usage |= BufferUsages::COPY_DST;
        let allocated_surfel_ids_stack =
            buffer("allocated_surfel_ids_stack", 4 * SURFEL_STACK_SIZE);
        let allocated_surfels_count = buffer("allocated_surfels_count", 4);
        let surfel_position = buffer("surfel_position", 12 * SURFEL_STACK_SIZE);
        let surfel_normal = buffer("surfel_normal", 12 * SURFEL_STACK_SIZE);
        let surfel_irradiance = buffer("surfel_irradiance", 12 * SURFEL_STACK_SIZE);

        commands.entity(entity).insert(SurfelsViewResources {
            unallocated_surfel_ids_stack: buffer_cache.get_or(
                &render_device,
                unallocated_surfel_ids_stack,
                &render_queue,
                || {
                    (0u32..SURFEL_STACK_SIZE as u32)
                        .flat_map(|v| v.to_le_bytes())
                        .collect()
                },
            ),
            allocated_surfel_ids_stack: buffer_cache
                .get(&render_device, allocated_surfel_ids_stack),
            allocated_surfels_count: buffer_cache.get(&render_device, allocated_surfels_count),
            surfel_position: buffer_cache.get(&render_device, surfel_position),
            surfel_normal: buffer_cache.get(&render_device, surfel_normal),
            surfel_irradiance: buffer_cache.get(&render_device, surfel_irradiance),
        });
    }
}

pub fn create_bind_group_layout(render_device: &RenderDevice) -> BindGroupLayout {
    let mut entry_i = 0;
    let mut entry = |ty| {
        entry_i += 1;
        BindGroupLayoutEntry {
            binding: entry_i - 1,
            visibility: ShaderStages::COMPUTE,
            ty,
            count: None,
        }
    };

    let entries = &[
        // View
        entry(BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(ViewUniform::min_size()),
        }),
        // unallocated_surfel_ids_stack
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
        }),
        // allocated_surfel_ids_stack
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
        }),
        // allocated_surfels_count
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
        }),
        // surfel_position
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(16) }),
        }),
        // surfel_normal
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(16) }),
        }),
        // surfel_irradiance
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(16) }),
        }),
    ];

    render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("surfels_view_bind_group_layout"),
        entries,
    })
}

#[derive(Component)]
pub struct SurfelsBindGroups {
    pub view_bind_group: BindGroup,
}

pub(crate) fn prepare_bind_groups(
    views: Query<(Entity, &SurfelsViewResources)>,
    view_uniforms: Res<ViewUniforms>,
    pipelines: Res<SurfelsPipelines>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
        return;
    };

    for (entity, surfels_res) in &views {
        let mut entry_i = 0;
        let mut entry = |resource| {
            entry_i += 1;
            BindGroupEntry {
                binding: entry_i - 1,
                resource,
            }
        };

        let entries = &[
            entry(view_uniforms.clone()),
            entry(b(&surfels_res.unallocated_surfel_ids_stack)),
            entry(b(&surfels_res.allocated_surfel_ids_stack)),
            entry(b(&surfels_res.allocated_surfels_count)),
            entry(b(&surfels_res.surfel_position)),
            entry(b(&surfels_res.surfel_normal)),
            entry(b(&surfels_res.surfel_irradiance)),
        ];

        let bind_groups = SurfelsBindGroups {
            view_bind_group: render_device.create_bind_group(&BindGroupDescriptor {
                label: Some("surfels_view_bind_group"),
                layout: &pipelines.view_bind_group_layout,
                entries,
            }),
        };
        commands.entity(entity).insert(bind_groups);
    }
}

fn b(buffer: &CachedBuffer) -> BindingResource<'_> {
    buffer.buffer.as_entire_binding()
}
