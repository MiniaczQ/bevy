use bevy_asset::AssetId;
use bevy_math::Vec3;
use bevy_render::{color::Color, render_resource::ShaderType, texture::Image};

pub struct GiMaterial {
    pub base_color: Color,
    pub base_color_texture: Option<AssetId<Image>>,
    pub normal_map_texture: Option<AssetId<Image>>,
    pub emissive: Color,
    pub emissive_texture: Option<AssetId<Image>>,
}

#[derive(ShaderType)]
pub struct GpuGiMaterial {
    pub base_color: [f32; 4],
    pub emissive: [f32; 4],
    pub base_color_texture_id: u32,
    pub normal_map_texture_id: u32,
    pub emissive_texture_id: u32,
    pub _padding: u32,
}

#[derive(ShaderType)]
pub struct DirectionalLight {
    pub direction_to_light: Vec3,
    pub color: [f32; 4],
}

#[derive(ShaderType)]
pub struct LightSource {
    kind: u32,
    id: u32,
}

impl LightSource {
    pub fn directional_light(id: u32) -> Self {
        Self { kind: u32::MAX, id }
    }

    pub fn emissive_triangle(object_id: u32, triangle_id: u32) -> Self {
        assert_ne!(triangle_id, u32::MAX);
        Self {
            kind: triangle_id,
            id: object_id,
        }
    }
}
