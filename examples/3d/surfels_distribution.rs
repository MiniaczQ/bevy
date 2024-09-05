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
            transform: Transform::from_translation(START_POS).looking_to(Vec3::Z, Vec3::Y),
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

/// Start frame for step and jump movement
const START_FRAME: u32 = 200;

/// Stop frame for step movement
const STOP_FRAME: u32 = 400;

const START_POS: Vec3 = Vec3::new(0.0, 1.5, -3.0);

const SWEEP_OFFSET: Vec3 = Vec3::new(0.71, 0.0, 0.0);

fn sweep(t: f32) -> Vec3 {
    START_POS + SWEEP_OFFSET * t
}

const ZOOM_OUT_OFFSET: Vec3 = Vec3::new(0.0, 0.0, -1.0);

fn zoom_out(t: f32) -> Vec3 {
    START_POS + ZOOM_OUT_OFFSET * t
}

const ZOOM_IN_OFFSET: Vec3 = Vec3::new(0.0, 0.0, 0.5);

fn zoom_in(t: f32) -> Vec3 {
    START_POS + ZOOM_IN_OFFSET * t
}

fn stay(_: f32) -> Vec3 {
    START_POS
}

fn step(frame: u32) -> f32 {
    if frame < START_FRAME {
        0.0
    } else if frame > STOP_FRAME {
        1.0
    } else {
        let length = (STOP_FRAME - START_FRAME) as f32;
        (frame - START_FRAME) as f32 / length
    }
}

fn jump(frame: u32) -> f32 {
    if frame < START_FRAME {
        0.0
    } else {
        1.0
    }
}

fn update(frame_count: Res<FrameCount>, mut camera: Query<&mut Transform, With<Camera>>) {
    let frame = frame_count.0;
    // Swap between `step` and `jump`
    let t = step(frame);
    // Swap between `sweep`, `zoom_in`, `zoom_out` and `stay`
    let pos = stay(t);
    camera.single_mut().translation = pos;
}
