use std::any::type_name;

use bevy_ecs::{
    entity::Entity,
    query::{QuerySingleError, With},
    system::Commands,
    world::{Command, World},
};
use bevy_utils::tracing::warn;

use crate::{data::StateData, state::State, GlobalStateMarker};

struct InsertGlobalState<S: State> {
    state: Option<S>,
    overwrite: bool,
}

impl<S: State> InsertGlobalState<S> {
    fn new(state: Option<S>, force: bool) -> Self {
        Self {
            state,
            overwrite: force,
        }
    }
}

impl<S: State + Send + Sync + 'static> Command for InsertGlobalState<S> {
    fn apply(self, world: &mut World) {
        // Register a global state entity
        let result = world
            .query_filtered::<Entity, With<GlobalStateMarker>>()
            .get_single(world);
        let entity = match result {
            Ok(entity) => entity,
            Err(QuerySingleError::NoEntities(_)) => world.spawn(GlobalStateMarker).id(),
            Err(QuerySingleError::MultipleEntities(_)) => {
                warn!(
                    "Insert global state command failed, multiple entities have the `GlobalStateMarker` component."
                );
                return;
            }
        };

        // Register storage for state `S`.
        let new_data = StateData::new(self.state);
        let state_data = world
            .query::<&mut StateData<S>>()
            .get_mut(world, entity)
            .ok();
        match (state_data, self.overwrite) {
            (Some(mut data), true) => {
                *data = new_data;
            }
            (None, _) => {
                world.entity_mut(entity).insert(new_data);
            }
            _ => {}
        }

        // Register observers for update.
        S::register_state(world);
    }
}

struct SetStateDeferred<S: State> {
    next: Option<S>,
    local: Option<Entity>,
}

impl<S: State> SetStateDeferred<S> {
    fn new(next: Option<S>, local: Option<Entity>) -> Self {
        Self { next, local }
    }
}

impl<S: State> Command for SetStateDeferred<S> {
    fn apply(self, world: &mut World) {
        let entity = match self.local {
            Some(entity) => entity,
            None => {
                match world
                    .query_filtered::<Entity, With<GlobalStateMarker>>()
                    .get_single(world)
                {
                    Err(QuerySingleError::NoEntities(_)) => {
                        warn!("Set global state command failed, no global state entity exists.");
                        return;
                    }
                    Err(QuerySingleError::MultipleEntities(_)) => {
                        warn!("Set global state command failed, multiple global state entities exist.");
                        return;
                    }
                    Ok(entity) => entity,
                }
            }
        };
        let Ok(mut state) = world.query::<&mut StateData<S>>().get_mut(world, entity) else {
            warn!(
                "Set state command failed, entity does not have state {}",
                type_name::<S>()
            );
            return;
        };
        state.next = Some(self.next);
    }
}

#[doc(hidden)]
pub trait CommandsExtStates {
    /// Inserts a global state.
    /// If `overwrite` is enabled, this will override the existing state.
    fn insert_global_state<S: State>(&mut self, value: Option<S>, overwrite: bool);

    /// Set the next value of the state.
    /// This value will be used to update the state in the [`StateTransition`] schedule.
    fn set_state<S: State>(&mut self, next: Option<S>, local: Option<Entity>);
}

impl CommandsExtStates for Commands<'_, '_> {
    fn insert_global_state<S: State>(&mut self, state: Option<S>, overwrite: bool) {
        self.add(InsertGlobalState::new(state, overwrite))
    }

    fn set_state<S: State>(&mut self, next: Option<S>, local: Option<Entity>) {
        self.add(SetStateDeferred::new(next, local))
    }
}

impl CommandsExtStates for World {
    fn insert_global_state<S: State>(&mut self, value: Option<S>, overwrite: bool) {
        self.commands().insert_global_state(value, overwrite);
    }

    fn set_state<S: State>(&mut self, next: Option<S>, local: Option<Entity>) {
        self.commands().set_state(next, local);
    }
}

#[cfg(feature = "bevy_app")]
impl CommandsExtStates for bevy_app::SubApp {
    fn insert_global_state<S: State>(&mut self, value: Option<S>, overwrite: bool) {
        self.world_mut().insert_global_state(value, overwrite);
    }

    fn set_state<S: State>(&mut self, next: Option<S>, local: Option<Entity>) {
        self.world_mut().set_state(next, local);
    }
}

#[cfg(feature = "bevy_app")]
impl CommandsExtStates for bevy_app::App {
    fn insert_global_state<S: State>(&mut self, value: Option<S>, overwrite: bool) {
        self.main_mut().insert_global_state(value, overwrite);
    }

    fn set_state<S: State>(&mut self, next: Option<S>, local: Option<Entity>) {
        self.main_mut().set_state(next, local);
    }
}
