use super::extract_asset_events::ExtractedAssetEvents;
use bevy_asset::AssetId;
use bevy_ecs::{
    system::{Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_render::{
    mesh::{GpuBufferInfo, GpuMesh, Mesh},
    render_asset::RenderAssets,
    render_resource::*,
    renderer::RenderDevice,
    texture::{FallbackImage, Image},
};
use bevy_utils::HashMap;
use std::{num::NonZeroU32, ops::Deref};

#[derive(Resource)]
pub struct AssetBindings {
    pub bind_group_layout: BindGroupLayout,
    pub mesh_indices: HashMap<AssetId<Mesh>, u32>,
    pub image_indices: HashMap<AssetId<Image>, u32>,
    pub bind_group: Option<BindGroup>,
}

impl FromWorld for AssetBindings {
    fn from_world(world: &mut World) -> Self {
        Self {
            bind_group_layout: world.resource::<RenderDevice>().create_bind_group_layout(
                "solari_assets_bind_group_layout",
                &bind_group_layout_entries(),
            ),
            mesh_indices: HashMap::new(),
            image_indices: HashMap::new(),
            bind_group: None,
        }
    }
}

pub fn prepare_asset_binding_arrays(
    mut asset_bindings: ResMut<AssetBindings>,
    asset_events: Res<ExtractedAssetEvents>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_images: Res<RenderAssets<Image>>,
    fallback_image: Res<FallbackImage>,
    render_device: Res<RenderDevice>,
) {
    if asset_events.meshes_changed.is_empty()
        && asset_events.images_changed.is_empty()
        && asset_bindings.bind_group.is_some()
    {
        return;
    }

    asset_bindings.image_indices.clear();
    asset_bindings.mesh_indices.clear();

    let (vertex_buffers, index_buffers) = render_meshes
        .iter()
        .filter(|(_, mesh)| mesh_solari_compatible(mesh))
        .enumerate()
        .map(|(i, (asset_id, mesh))| {
            asset_bindings.mesh_indices.insert(asset_id, i as u32);
            let index_buffer = match &mesh.buffer_info {
                GpuBufferInfo::Indexed { buffer, .. } => buffer,
                GpuBufferInfo::NonIndexed => unreachable!(),
            };
            (
                mesh.vertex_buffer.as_entire_buffer_binding(),
                index_buffer.as_entire_buffer_binding(),
            )
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    if vertex_buffers.is_empty() {
        return;
    }

    let device_features = Some(render_device.features());
    let (mut images, mut samplers) = render_images
        .iter()
        .filter(|(_, image)| {
            image.texture_format.sample_type(None, device_features)
                == Some(TextureSampleType::Float { filterable: true })
                && image.texture.dimension() == TextureDimension::D2
                && image.texture.sample_count() == 1
        })
        .enumerate()
        .map(|(i, (asset_id, image))| {
            asset_bindings.image_indices.insert(asset_id, i as u32);
            (image.texture_view.deref(), image.sampler.deref())
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();
    images.push(&fallback_image.d2.texture_view);
    samplers.push(&fallback_image.d2.sampler);

    asset_bindings.bind_group = Some(render_device.create_bind_group(
        "solari_assets_bind_group",
        &asset_bindings.bind_group_layout,
        &BindGroupEntries::sequential((
            vertex_buffers.as_slice(),
            index_buffers.as_slice(),
            images.as_slice(),
            samplers.as_slice(),
        )),
    ));
}

pub fn mesh_solari_compatible(mesh: &GpuMesh) -> bool {
    let triangle_list = mesh.primitive_topology == PrimitiveTopology::TriangleList;
    let vertex_layout = mesh.layout.attribute_ids()
        == &[
            Mesh::ATTRIBUTE_POSITION.id,
            Mesh::ATTRIBUTE_NORMAL.id,
            Mesh::ATTRIBUTE_UV_0.id,
            Mesh::ATTRIBUTE_TANGENT.id,
        ];
    let indexed_32 = matches!(
        mesh.buffer_info,
        GpuBufferInfo::Indexed {
            index_format: IndexFormat::Uint32,
            ..
        }
    );
    triangle_list && vertex_layout && indexed_32 && mesh.ray_tracing_support
}

// TODO: Configurable max resources
fn bind_group_layout_entries() -> [BindGroupLayoutEntry; 4] {
    [
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None, // TODO
            },
            count: NonZeroU32::new(1000),
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None, // TODO
            },
            count: NonZeroU32::new(1000),
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: NonZeroU32::new(1000),
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: NonZeroU32::new(1000),
        },
    ]
}
