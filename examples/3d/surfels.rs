//! Demonstrates realtime dynamic global illumination rendering using Bevy Solari.

#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use bevy::{
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass},
    pbr::global_illumination::{
        GlobalIlluminationPlugin, GlobalIlluminationSettings, GlobalIlluminationSupported,
    },
    prelude::*,
    render::camera::CameraMainTextureUsages,
};
use camera_controller::{CameraController, CameraControllerPlugin};
use std::f32::consts::PI;

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
                solari_not_supported.run_if(not(resource_exists::<GlobalIlluminationSupported>)),
                setup.run_if(resource_exists::<GlobalIlluminationSupported>),
            ),
        )
        .add_systems(
            Update,
            toggle_solari.run_if(resource_exists::<GlobalIlluminationSupported>),
        )
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/CornellBox/box_modified.glb#Scene0"),
        ..default()
    });

    //commands.spawn(DirectionalLightBundle {
    //    directional_light: DirectionalLight {
    //        shadows_enabled: true,
    //        ..default()
    //    },
    //    transform: Transform::from_rotation(Quat::from_euler(
    //        EulerRot::XYZ,
    //        PI * -0.43,
    //        PI * -0.08,
    //        0.0,
    //    )),
    //    ..default()
    //});

    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 1.5, 5.0).looking_to(-Vec3::Z, Vec3::Y),
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

fn solari_not_supported(mut commands: Commands) {
    commands.spawn(
        TextBundle::from_section(
            "Current GPU does not support Solari",
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

fn toggle_solari(
    key_input: Res<ButtonInput<KeyCode>>,
    mut commands: Commands,
    camera: Query<(Entity, Has<GlobalIlluminationSettings>), With<Camera>>,
) {
    if key_input.just_pressed(KeyCode::Space) {
        let (entity, solari_enabled) = camera.single();
        if solari_enabled {
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
