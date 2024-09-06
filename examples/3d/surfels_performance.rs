//! Demonstrates realtime dynamic global illumination rendering using Bevy Solari.

#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use bevy::{
    core::FrameCount,
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass},
    pbr::global_illumination::{
        GlobalIlluminationPlugin, GlobalIlluminationSettings, GlobalIlluminationSupported,
    },
    prelude::*,
    render::camera::CameraMainTextureUsages,
};
use camera_controller::CameraController;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, GlobalIlluminationPlugin))
        .add_systems(
            Startup,
            (
                not_supported.run_if(not(resource_exists::<GlobalIlluminationSupported>)),
                setup.run_if(resource_exists::<GlobalIlluminationSupported>),
            ),
        )
        .add_systems(Update, update)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/CornellBox/box_modified.glb#Scene0"),
        ..default()
    });

    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            transform: Transform::from_translation(CENTER).looking_to(-Vec3::Z, Vec3::Y),
            main_texture_usages: CameraMainTextureUsages::with_storage_binding(),
            ..default()
        },
        DeferredPrepass,
        DepthPrepass,
        MotionVectorPrepass,
        GlobalIlluminationSettings::default(),
        CameraController::default(),
    ));
}

fn not_supported(mut commands: Commands) {
    commands.spawn(
        TextBundle::from_section(
            "Current GPU does not support Ray Tracing",
            TextStyle {
                font_size: 48.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            margin: UiRect::all(Val::Auto),
            ..default()
        }),
    );

    commands.spawn(Camera2dBundle::default());
}

const CENTER: Vec3 = Vec3::new(0.0, 1.5, 1.5);
const RADIUS: f32 = 0.75;
const SPEED: f32 = std::f32::consts::TAU;

fn update(time: Res<Time<Virtual>>, mut camera: Query<&mut Transform, With<Camera>>) {
    let t = time.elapsed_seconds_f64() as f32 * SPEED;
    let (sin, cos) = t.sin_cos();
    let offset = Vec3::new(sin, cos, 0.0) * RADIUS;
    let pos = CENTER + offset;
    camera.single_mut().translation = pos;
}
