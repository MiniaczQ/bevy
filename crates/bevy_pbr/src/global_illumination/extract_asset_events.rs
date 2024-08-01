use super::gpu_types::GiMaterial;
use crate::StandardMaterial;
use bevy_asset::{AssetEvent, AssetId, Assets, Handle};
use bevy_ecs::{
    event::EventReader,
    system::{ResMut, Resource, SystemState},
    world::{FromWorld, Mut, World},
};
use bevy_render::{mesh::Mesh, texture::Image, MainWorld};
use bevy_utils::{HashMap, HashSet};

#[derive(Resource, Default)]
pub struct ExtractedAssetEvents {
    pub meshes_changed: HashSet<AssetId<Mesh>>,
    pub meshes_removed: Vec<AssetId<Mesh>>,
    pub images_changed: HashSet<AssetId<Image>>,
    pub images_removed: Vec<AssetId<Image>>,
    materials_changed: HashSet<AssetId<StandardMaterial>>,
    materials_removed: Vec<AssetId<StandardMaterial>>,
    pub materials: HashMap<AssetId<StandardMaterial>, GiMaterial>,
}

pub fn extract_asset_events(
    mut main_world: ResMut<MainWorld>,
    mut asset_events: ResMut<ExtractedAssetEvents>,
) {
    let asset_events = asset_events.as_mut();

    asset_events.meshes_changed.clear();
    asset_events.meshes_removed.clear();

    asset_events.images_changed.clear();
    asset_events.images_removed.clear();

    main_world.resource_scope(
        |main_world, mut state: Mut<ExtractAssetEventsSystemState>| {
            let (mut mesh_events, mut image_events, mut material_events) =
                state.state.get(main_world);

            for asset_event in mesh_events.read() {
                match asset_event {
                    AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                        asset_events.meshes_changed.insert(*id);
                    }
                    AssetEvent::Unused { id } => {
                        asset_events.meshes_removed.push(*id);
                        asset_events.meshes_changed.remove(id);
                    }
                    _ => {}
                }
            }

            for asset_event in image_events.read() {
                match asset_event {
                    AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                        asset_events.images_changed.insert(*id);
                    }
                    AssetEvent::Unused { id } => {
                        asset_events.images_removed.push(*id);
                        asset_events.images_changed.remove(id);
                    }
                    _ => {}
                }
            }

            for asset_event in material_events.read() {
                match asset_event {
                    AssetEvent::Added { id } | AssetEvent::Modified { id } => {
                        asset_events.materials_changed.insert(*id);
                    }
                    AssetEvent::Unused { id } => {
                        asset_events.materials_removed.push(*id);
                        asset_events.materials_changed.remove(id);
                    }
                    _ => {}
                }
            }
        },
    );

    for asset_id in asset_events.materials_removed.drain(..) {
        asset_events.materials.remove(&asset_id);
    }

    let materials = main_world.resource::<Assets<StandardMaterial>>();
    for asset_id in asset_events.materials_changed.drain() {
        if let Some(material) = materials.get(asset_id) {
            let solari_material = GiMaterial {
                base_color: material.base_color,
                base_color_texture: material.base_color_texture.as_ref().map(Handle::id),
                normal_map_texture: material.normal_map_texture.as_ref().map(Handle::id),
                emissive: material.emissive,
                emissive_texture: material.emissive_texture.as_ref().map(Handle::id),
            };
            asset_events.materials.insert(asset_id, solari_material);
        }
    }
}

#[derive(Resource)]
pub struct ExtractAssetEventsSystemState {
    state: SystemState<(
        EventReader<'static, 'static, AssetEvent<Mesh>>,
        EventReader<'static, 'static, AssetEvent<Image>>,
        EventReader<'static, 'static, AssetEvent<StandardMaterial>>,
    )>,
}

impl FromWorld for ExtractAssetEventsSystemState {
    fn from_world(world: &mut World) -> Self {
        Self {
            state: SystemState::new(world),
        }
    }
}
