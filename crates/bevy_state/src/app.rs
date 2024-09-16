use bevy_app::{MainScheduleOrder, Plugin, PreStartup, PreUpdate};

use crate::scheduling::StateTransition;

#[derive(Default)]
pub struct StatePlugin;

impl Plugin for StatePlugin {
    fn build(&self, app: &mut bevy_app::App) {
        let mut schedule = app.world_mut().resource_mut::<MainScheduleOrder>();
        schedule.insert_startup_before(PreStartup, StateTransition);
        schedule.insert_after(PreUpdate, StateTransition);
    }
}
