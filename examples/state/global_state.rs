//! This example shows how to use the most basic global state machine.
//! The machine consists of a single state type that decides
//! whether a logo moves around the screen and changes color.

use bevy::{prelude::*, sprite::Anchor};
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Register machinery for the state.
        // This is required for both global and local state, but only needs to be called once.
        // By providing an empty config we opt-out of state transition events.
        .register_state::<LogoState>(StateTransitionsConfig::empty())
        // By targeting no specific entity, we create a global state.
        // We provide the initial state value.
        // Because we're not using transition events or state hierarchy, update suppresion doesn't matter.
        .init_state(None, Some(LogoState::Enabled), true)
        .add_systems(Startup, setup)
        .add_systems(Update, toggle_logo)
        .add_systems(
            Update,
            // We can use global state to determine when certain systems run.
            (bounce_around, cycle_color).run_if(in_state(Some(LogoState::Enabled))),
        )
        .run();
}

#[derive(State, PartialEq, Debug, Clone)]
enum LogoState {
    Enabled,
    Disabled,
}

/// When we click `1` on the keyboard, the logo activity will be toggled.
fn toggle_logo(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    state: GlobalState<LogoState>,
) {
    if input.just_pressed(KeyCode::Digit1) {
        // Decide the next state based on current state.
        let next = match state.get().current().unwrap() {
            LogoState::Enabled => LogoState::Disabled,
            LogoState::Disabled => LogoState::Enabled,
        };
        // Request a change for the state.
        commands.state_target(None, Some(next));
    }
}

/// Half of the logo size for collision checking.
const LOGO_HALF_SIZE: Vec2 = Vec2::new(260., 65.);

/// Where the logo is going.
#[derive(Component)]
struct Velocity(Vec2);

/// Create the camera and logo.
fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    // Add camera.
    commands.spawn(Camera2dBundle::default());

    // Create logo with random position and velocity.
    let mut rng = rand::thread_rng();
    let texture = assets.load("branding/bevy_logo_dark.png");
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::hsv(rng.gen_range(0.0..=1.0), 1.0, 1.0),
                anchor: Anchor::Center,
                ..default()
            },
            texture,
            transform: Transform::from_xyz(
                rng.gen_range(-200.0..=200.),
                rng.gen_range(-200.0..=200.),
                0.,
            ),
            ..default()
        },
        Velocity(Dir2::from_rng(&mut rng) * rng.gen_range(0.0..=10.)),
    ));
}

/// Make the logo bounce.
fn bounce_around(
    mut logos: Query<(&mut Transform, &mut Velocity), With<Sprite>>,
    camera: Query<&OrthographicProjection>,
) {
    let camera = camera.single();
    for (mut transform, mut velocity) in logos.iter_mut() {
        transform.translation += velocity.0.extend(0.);
        let logo_pos = transform.translation.xy();

        let mut flip_x = false;
        let x_max = camera.area.max.x - LOGO_HALF_SIZE.x;
        if x_max < logo_pos.x {
            transform.translation.x = x_max;
            flip_x = !flip_x;
        }
        let x_min = camera.area.min.x + LOGO_HALF_SIZE.x;
        if logo_pos.x < x_min {
            transform.translation.x = x_min;
            flip_x = !flip_x;
        }
        if flip_x {
            velocity.0.x *= -1.;
        }

        let mut flip_y = false;
        let y_max = camera.area.max.y - LOGO_HALF_SIZE.y;
        if y_max < logo_pos.y {
            transform.translation.y = y_max;
            flip_y = !flip_y;
        }
        let y_min = camera.area.min.y + LOGO_HALF_SIZE.y;
        if logo_pos.y < y_min {
            transform.translation.y = y_min;
            flip_y = !flip_y;
        }
        if flip_y {
            velocity.0.y *= -1.;
        }
    }
}

/// Make the logo rainbow.
fn cycle_color(mut logos: Query<&mut Sprite, With<Sprite>>) {
    for mut sprite in logos.iter_mut() {
        sprite.color = sprite.color.rotate_hue(0.3);
    }
}

// example ideas:
// - target backends: returning substate, toggle, force retransition
// - local state: 2 logos
