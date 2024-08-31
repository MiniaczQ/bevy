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
    render::{camera::CameraMainTextureUsages, mesh::PlaneMeshBuilder},
};
use camera_controller::{CameraController, CameraControllerPlugin};
use rand::distributions::Standard;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            GlobalIlluminationPlugin,
            CameraControllerPlugin,
        ))
        .add_systems(
            Startup,
            (
                not_supported.run_if(not(resource_exists::<GlobalIlluminationSupported>)),
                setup.run_if(resource_exists::<GlobalIlluminationSupported>),
            ),
        )
        .add_systems(
            Update,
            toggle.run_if(resource_exists::<GlobalIlluminationSupported>),
        )
        .add_systems(Update, update)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(PlaneMeshBuilder::new(Direction3d::Z, Vec2::new(1000.0, 1000.0)).build());
    let material = materials.add(Color::WHITE);
    commands.spawn(MaterialMeshBundle {
        mesh,
        material,
        ..Default::default()
    });

    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 0.0, 5.0).looking_to(-Vec3::Z, Vec3::Y),
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

const START_POS: Vec3 = Vec3::new(0.0, 0.0, 5.0);

const SWEEP_OFFSET: Vec3 = Vec3::new(5.0, 0.0, 0.0);

fn sweep(t: f32) -> Vec3 {
    START_POS + SWEEP_OFFSET * t
}

const ZOOM_OUT_OFFSET: Vec3 = Vec3::new(0.0, 0.0, 5.0);

fn zoom_out(t: f32) -> Vec3 {
    START_POS + ZOOM_OUT_OFFSET * t
}

const ZOOM_IN_OFFSET: Vec3 = Vec3::new(0.0, 0.0, -2.5);

fn zoom_in(t: f32) -> Vec3 {
    START_POS + ZOOM_IN_OFFSET * t
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
    let t = step(frame);
    let pos = sweep(t);
    camera.single_mut().translation = pos;
}

fn toggle(
    key_input: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    camera: Query<(Entity, Has<GlobalIlluminationSettings>), With<Camera>>,
) {
    if key_input.just_pressed(KeyCode::Space) {
        let (entity, enabled) = camera.single();
        if enabled {
            commands
                .entity(entity)
                .remove::<GlobalIlluminationSettings>();
        } else {
            commands
                .entity(entity)
                .insert(GlobalIlluminationSettings::default());
        }
    }
}
