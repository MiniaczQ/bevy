use super::{asset_binder::mesh_solari_compatible, extract_asset_events::ExtractedAssetEvents};
use bevy_asset::AssetId;
use bevy_ecs::system::{Res, ResMut, Resource};
use bevy_render::{
    mesh::{GpuBufferInfo, GpuMesh, Mesh},
    render_asset::RenderAssets,
    render_resource::*,
    renderer::{RenderDevice, RenderQueue},
};
use bevy_utils::HashMap;

#[derive(Resource, Default)]
pub struct BlasManager {
    info: HashMap<AssetId<Mesh>, (Blas, u32)>,
}

impl BlasManager {
    pub fn get_blas_and_triangle_count(&self, mesh: &AssetId<Mesh>) -> Option<&(Blas, u32)> {
        self.info.get(mesh)
    }
}

// TODO: BLAS compaction
// TODO: Async compute queue for BLAS creation
pub fn prepare_new_blas(
    mut blas_manager: ResMut<BlasManager>,
    asset_events: Res<ExtractedAssetEvents>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    let blas_manager = blas_manager.as_mut();

    // Delete BLAS for removed meshes
    for asset_id in &asset_events.meshes_removed {
        blas_manager.info.remove(asset_id);
    }

    if asset_events.meshes_changed.is_empty() {
        return;
    }

    // Get GpuMeshes and filter to solari-compatible meshes
    let meshes = asset_events
        .meshes_changed
        .iter()
        .filter_map(|asset_id| match render_meshes.get(*asset_id) {
            Some(mesh) if mesh_solari_compatible(mesh) => Some((asset_id, mesh)),
            _ => None,
        })
        .collect::<Vec<_>>();

    // Create BLAS, blas size for each mesh
    let blas_resources = meshes
        .iter()
        .map(|(asset_id, mesh)| setup_blas(asset_id, mesh, blas_manager, &render_device))
        .collect::<Vec<_>>();

    // Create list of BlasBuildEntries using blas_resources
    let build_entries = blas_resources
        .iter()
        .map(|(asset_id, mesh, blas_size, index_buffer)| BlasBuildEntry {
            blas: &blas_manager.info.get(*asset_id).unwrap().0,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &mesh.vertex_buffer,
                first_vertex: 0,
                vertex_stride: mesh.layout.layout().array_stride,
                index_buffer: Some(index_buffer),
                index_buffer_offset: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect::<Vec<_>>();

    // Build geometry into each BLAS
    let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("build_blas_command_encoder"),
    });
    command_encoder.build_acceleration_structures(&build_entries, &[]);
    render_queue.submit([command_encoder.finish()]);
}

fn setup_blas<'a, 'b>(
    asset_id: &'a AssetId<Mesh>,
    mesh: &'a GpuMesh,
    blas_manager: &'b mut BlasManager,
    render_device: &'b RenderDevice,
) -> (
    &'a AssetId<Mesh>,
    &'a GpuMesh,
    BlasTriangleGeometrySizeDescriptor,
    &'a Buffer,
) {
    let (index_buffer, index_count) = {
        match &mesh.buffer_info {
            GpuBufferInfo::Indexed { buffer, count, .. } => (buffer, *count),
            GpuBufferInfo::NonIndexed => unreachable!(),
        }
    };

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: Mesh::ATTRIBUTE_POSITION.format,
        vertex_count: mesh.vertex_count,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(index_count),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };

    let blas = render_device.wgpu_device().create_blas(
        &CreateBlasDescriptor {
            label: Some("blas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            desc: vec![blas_size.clone()],
        },
    );
    blas_manager.info.insert(*asset_id, (blas, index_count / 3));

    (asset_id, mesh, blas_size, index_buffer)
}
