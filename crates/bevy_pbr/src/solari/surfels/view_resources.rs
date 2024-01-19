use super::{SurfelsPipelines, SurfelsSettings, MAX_SURFELS};
use bevy_core_pipeline::prepass::{
    DepthPrepass, MotionVectorPrepass, NormalPrepass, ViewPrepassTextures,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferCache,
        BufferDescriptor, BufferUsages, CachedBuffer, Extent3d, ShaderStages, ShaderType,
        StorageTextureAccess, TextureDescriptor, TextureDimension, TextureFormat,
        TextureSampleType, TextureUsages, TextureViewDimension,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::{ViewUniform, ViewUniforms},
};
use std::num::NonZeroU64;

#[derive(Component)]
pub struct SurfelsViewResources {
    unallocated_surfel_ids_stack: CachedBuffer,
    allocated_surfels_bitmap: CachedBuffer,
    allocated_surfel_ids_count: CachedBuffer,
    surfel_position: CachedBuffer,
    surfel_normal: CachedBuffer,
    surfel_irradiance: CachedBuffer,
    pub diffuse_irradiance_output: CachedTexture,
    pub surfels_to_allocate: CachedBuffer,
    surfel_grid_allocate: CachedBuffer,
}

pub fn prepare_resources(
    views: Query<
        (Entity, &ExtractedCamera),
        (
            With<SurfelsSettings>,
            With<DepthPrepass>,
            With<NormalPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    mut commands: Commands,
    mut buffer_cache: ResMut<BufferCache>,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let texture = |label, format, size: UVec2| TextureDescriptor {
        label: Some(label),
        size: Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    };
    let buffer = |label, size| BufferDescriptor {
        label: Some(label),
        size,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    };

    for (entity, camera) in &views {
        let Some(viewport_size) = camera.physical_viewport_size else {
            continue;
        };
        let mut unallocated_surfel_ids_stack =
            buffer("unallocated_surfel_ids_stack", 4 * MAX_SURFELS);
        unallocated_surfel_ids_stack.usage |= BufferUsages::COPY_DST;
        let allocated_surfels_bitmap = buffer("allocated_surfels_bitmap", 4 * MAX_SURFELS / 32);
        let allocated_surfel_ids_count = buffer("allocated_surfel_ids_count", 4);
        let surfel_position = buffer("surfel_position", 16 * MAX_SURFELS);
        let surfel_normal = buffer("surfel_normal", 16 * MAX_SURFELS);
        let surfel_irradiance = buffer("surfel_irradiance", 16 * MAX_SURFELS);
        let mut diffuse_irradiance_output = texture(
            "diffuse_irradiance_output",
            TextureFormat::Rgba16Float,
            viewport_size,
        );
        diffuse_irradiance_output.usage |= TextureUsages::TEXTURE_BINDING;
        let surfel_grid_allocate = buffer("surfel_grid_allocate", 4 * 16 * 16);
        let mut surfels_to_allocate = buffer("surfels_to_allocate", 12);
        surfels_to_allocate.usage |= BufferUsages::INDIRECT;

        commands.entity(entity).insert(SurfelsViewResources {
            unallocated_surfel_ids_stack: buffer_cache.get_or(
                &render_device,
                unallocated_surfel_ids_stack,
                &render_queue,
                || {
                    (0u32..MAX_SURFELS as u32)
                        .flat_map(|v| v.to_le_bytes())
                        .collect()
                },
            ),
            allocated_surfels_bitmap: buffer_cache.get(&render_device, allocated_surfels_bitmap),
            allocated_surfel_ids_count: buffer_cache
                .get(&render_device, allocated_surfel_ids_count),
            surfel_position: buffer_cache.get(&render_device, surfel_position),
            surfel_normal: buffer_cache.get(&render_device, surfel_normal),
            surfel_irradiance: buffer_cache.get(&render_device, surfel_irradiance),
            diffuse_irradiance_output: texture_cache.get(&render_device, diffuse_irradiance_output),
            surfel_grid_allocate: buffer_cache.get(&render_device, surfel_grid_allocate),
            surfels_to_allocate: buffer_cache.get(&render_device, surfels_to_allocate),
        });
    }
}

pub fn create_bind_group_layout(
    render_device: &RenderDevice,
) -> (BindGroupLayout, BindGroupLayout) {
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
        // Depth buffer
        entry(BindingType::Texture {
            sample_type: TextureSampleType::Depth,
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }),
        // Normals buffer
        entry(BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: false },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        }),
        // unallocated_surfel_ids_stack
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
        }),
        // allocated_surfels_bitmap
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4) }),
        }),
        // allocated_surfel_ids_count
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
        // diffuse_irradiance_output
        entry(BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format: TextureFormat::Rgba16Float,
            view_dimension: TextureViewDimension::D2,
        }),
        // surfel_grid_allocate
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(4 * 16 * 16) }),
        }),
        // surfels_to_allocate
        entry(BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: Some(unsafe { NonZeroU64::new_unchecked(12) }),
        }),
    ];

    (
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("surfels_view_bind_group_layout_with_surfels_to_allocate"),
            entries,
        }),
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("surfels_view_bind_group_layout"),
            entries: &entries[0..entries.len() - 1],
        }),
    )
}

#[derive(Component)]
pub struct SurfelsBindGroups {
    pub view_bind_group: BindGroup,
    pub view_bind_group_with_surfels_to_allocate: BindGroup,
}

pub(crate) fn prepare_bind_groups(
    views: Query<(Entity, &SurfelsViewResources, &ViewPrepassTextures)>,
    view_uniforms: Res<ViewUniforms>,
    pipelines: Res<SurfelsPipelines>,
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    let Some(view_uniforms) = view_uniforms.uniforms.binding() else {
        return;
    };

    for (entity, surfels_res, prepass_textures) in &views {
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
            entry(t(prepass_textures.depth.as_ref().unwrap())),
            entry(t(prepass_textures.normal.as_ref().unwrap())),
            entry(b(&surfels_res.unallocated_surfel_ids_stack)),
            entry(b(&surfels_res.allocated_surfels_bitmap)),
            entry(b(&surfels_res.allocated_surfel_ids_count)),
            entry(b(&surfels_res.surfel_position)),
            entry(b(&surfels_res.surfel_normal)),
            entry(b(&surfels_res.surfel_irradiance)),
            entry(t(&surfels_res.diffuse_irradiance_output)),
            entry(b(&surfels_res.surfel_grid_allocate)),
            entry(b(&surfels_res.surfels_to_allocate)),
        ];

        let bind_groups = SurfelsBindGroups {
            view_bind_group_with_surfels_to_allocate: render_device.create_bind_group(
                &BindGroupDescriptor {
                    label: Some("surfels_view_bind_group_with_surfels_to_allocate"),
                    layout: &pipelines.view_bind_group_layout_with_surfels_to_allocate,
                    entries,
                },
            ),
            view_bind_group: render_device.create_bind_group(&BindGroupDescriptor {
                label: Some("surfels_view_bind_group"),
                layout: &pipelines.view_bind_group_layout,
                entries: &entries[0..entries.len() - 1],
            }),
        };
        commands.entity(entity).insert(bind_groups);
    }
}

fn t(texture: &CachedTexture) -> BindingResource<'_> {
    BindingResource::TextureView(&texture.default_view)
}

fn b(buffer: &CachedBuffer) -> BindingResource<'_> {
    buffer.buffer.as_entire_binding()
}
