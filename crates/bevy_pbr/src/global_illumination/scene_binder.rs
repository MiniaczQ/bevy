use super::{
    asset_binder::AssetBindings,
    blas_manager::BlasManager,
    extract_asset_events::ExtractedAssetEvents,
    gpu_types::{DirectionalLight, GpuGiMaterial, LightSource},
};
use crate::{ExtractedDirectionalLight, StandardMaterial};
use bevy_asset::{AssetId, Handle};
use bevy_ecs::{
    system::{Query, Res, ResMut, Resource},
    world::{FromWorld, World},
};
use bevy_math::Mat4;
use bevy_render::{
    mesh::Mesh,
    render_resource::{binding_types::storage_buffer_read_only, encase::internal::WriteInto, *},
    renderer::{RenderDevice, RenderQueue},
    texture::Image,
    Extract,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::HashMap;
use std::iter;

#[derive(Resource, Default)]
pub struct ExtractedScene {
    entities: Vec<(AssetId<Mesh>, AssetId<StandardMaterial>, GlobalTransform)>,
}

pub fn extract_scene(
    mut scene: ResMut<ExtractedScene>,
    query: Extract<Query<(&Handle<Mesh>, &Handle<StandardMaterial>, &GlobalTransform)>>,
) {
    scene.entities.clear();

    for (mesh_handle, material_handle, transform) in &query {
        scene
            .entities
            .push((mesh_handle.id(), material_handle.id(), transform.clone()));
    }
}

#[derive(Resource)]
pub struct SceneBindings {
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: Option<BindGroup>,
    // TODO: Needed for now because the bind group isin't properly keeping the tlas alive
    tlas: Option<TlasPackage>,
}

impl FromWorld for SceneBindings {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self {
            bind_group_layout: render_device.create_bind_group_layout(
                "solari_scene_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        BindingType::AccelerationStructure,
                        storage_buffer_read_only::<u32>(false),
                        storage_buffer_read_only::<Mat4>(false),
                        storage_buffer_read_only::<GpuGiMaterial>(false),
                        storage_buffer_read_only::<LightSource>(false),
                        storage_buffer_read_only::<DirectionalLight>(false),
                    ),
                ),
            ),
            bind_group: None,
            tlas: None,
        }
    }
}

// TODO: Optimize buffer management
pub fn prepare_scene_bindings(
    mut scene_bindings: ResMut<SceneBindings>,
    directional_lights_query: Query<&ExtractedDirectionalLight>,
    asset_bindings: Res<AssetBindings>,
    scene: Res<ExtractedScene>,
    asset_events: Res<ExtractedAssetEvents>,
    blas_manager: Res<BlasManager>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // Build buffer of materials
    let get_image_id = |asset_id: Option<AssetId<Image>>| match asset_id {
        Some(asset_id) => *asset_bindings
            .image_indices
            .get(&asset_id)
            .unwrap_or(&u32::MAX),
        None => u32::MAX,
    };
    let mut materials = Vec::with_capacity(asset_events.materials.len());
    let mut material_ids = HashMap::with_capacity(asset_events.materials.len());
    for (asset_id, material) in &asset_events.materials {
        material_ids.insert(*asset_id, materials.len() as u32);
        materials.push(GpuGiMaterial {
            base_color: material.base_color.as_linear_rgba_f32(),
            emissive: material.emissive.as_linear_rgba_f32(),
            base_color_texture_id: get_image_id(material.base_color_texture),
            normal_map_texture_id: get_image_id(material.normal_map_texture),
            emissive_texture_id: get_image_id(material.emissive_texture),
            _padding: 0,
        });
    }

    let mut light_sources = Vec::new();

    // Build buffer of directional lights
    let mut directional_lights = Vec::new();
    for directional_light in &directional_lights_query {
        light_sources.push(LightSource::directional_light(
            directional_lights.len() as u32
        ));
        directional_lights.push(DirectionalLight {
            direction_to_light: directional_light.transform.back(),
            color: (directional_light.color.as_rgba_linear() * directional_light.illuminance)
                .as_linear_rgba_f32(),
        });
    }

    // Create TLAS
    let mut tlas = TlasPackage::new(
        render_device
            .wgpu_device()
            .create_tlas(&CreateTlasDescriptor {
                label: Some("tlas"),
                flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: AccelerationStructureUpdateMode::Build,
                max_instances: scene.entities.len() as u32,
            }),
        scene.entities.len() as u32,
    );

    // Build each entity into the TLAS and push its transform/mesh_id/material_id to a GPU buffer
    let mut object_id = 0;
    let mut transforms = Vec::with_capacity(scene.entities.len());
    let mut mesh_material_ids = Vec::with_capacity(scene.entities.len());
    for (mesh_id, material_id, transform) in &scene.entities {
        if let (Some((blas, triangle_count)), Some(mesh_id), Some(material_id)) = (
            blas_manager.get_blas_and_triangle_count(mesh_id),
            asset_bindings.mesh_indices.get(mesh_id),
            material_ids.get(material_id),
        ) {
            let transform = transform.compute_matrix();
            transforms.push(transform);

            // TODO: Check for ID overflow
            mesh_material_ids.push((*mesh_id << 16) | *material_id);

            // For emissive meshes, push each triangle to the light sources buffer
            let material = &materials[*material_id as usize];
            if material.emissive != [0.0; 4] || material.emissive_texture_id != u32::MAX {
                for triangle_id in 0..*triangle_count {
                    light_sources.push(LightSource::emissive_triangle(object_id, triangle_id));
                }
            }

            *tlas.get_mut_single(object_id as usize).unwrap() = Some(TlasInstance::new(
                blas,
                tlas_transform(&transform),
                object_id, // TODO: Max 24 bits
                0xFF,
            ));

            object_id += 1;
        }
    }

    // Build TLAS
    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_tlas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&[], iter::once(&tlas));
    render_queue.submit([command_encoder.finish()]);

    // Upload GPU buffers
    let mesh_material_ids = &new_storage_buffer(
        mesh_material_ids,
        "solari_mesh_material_ids",
        &render_device,
        &render_queue,
    );
    let transforms = new_storage_buffer(
        transforms,
        "solari_transforms",
        &render_device,
        &render_queue,
    );
    let materials =
        new_storage_buffer(materials, "solari_materials", &render_device, &render_queue);
    let light_sources = new_storage_buffer(
        light_sources,
        "solari_light_sources",
        &render_device,
        &render_queue,
    );
    let directional_lights = new_storage_buffer(
        directional_lights,
        "solari_directional_lights",
        &render_device,
        &render_queue,
    );

    // Create a bind group for the created resources
    scene_bindings.bind_group = Some(render_device.create_bind_group(
        "solari_scene_bind_group",
        &scene_bindings.bind_group_layout,
        &BindGroupEntries::sequential((
            tlas.as_binding(),
            mesh_material_ids.binding().unwrap(),
            transforms.binding().unwrap(),
            materials.binding().unwrap(),
            light_sources.binding().unwrap(),
            directional_lights.binding().unwrap(),
        )),
    ));
    scene_bindings.tlas = Some(tlas);
}

fn new_storage_buffer<T: ShaderSize + WriteInto>(
    vec: Vec<T>,
    label: &'static str,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) -> StorageBuffer<Vec<T>> {
    let mut buffer = StorageBuffer::from(vec);
    buffer.set_label(Some(label));
    buffer.write_buffer(render_device, render_queue);
    buffer
}

fn tlas_transform(transform: &Mat4) -> [f32; 12] {
    transform.transpose().to_cols_array()[..12]
        .try_into()
        .unwrap()
}
