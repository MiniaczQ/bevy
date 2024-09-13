use std::any::type_name;

use bevy_ecs::{
    entity::Entity,
    query::{QuerySingleError, With},
    system::Commands,
    world::{Command, World},
};
use bevy_utils::tracing::warn;

use crate::{data::StateData, state::{GlobalStateMarker, State}};

struct InsertStateCommand<S: State> {
    local: Option<Entity>,
    next: Option<S>,
    overwrite: bool,
}

impl<S: State> InsertStateCommand<S> {
    fn new(local: Option<Entity>, next: Option<S>, overwrite: bool) -> Self {
        Self {
            local,
            next,
            overwrite,
        }
    }
}

impl<S: State + Send + Sync + 'static> Command for InsertStateCommand<S> {
    fn apply(self, world: &mut World) {
        let entity = match self.local {
            Some(entity) => entity,
            None => {
                let result = world
                    .query_filtered::<Entity, With<GlobalStateMarker>>()
                    .get_single(world);
                match result {
                    Ok(entity) => entity,
                    Err(QuerySingleError::NoEntities(_)) => world.spawn(GlobalStateMarker).id(),
                    Err(QuerySingleError::MultipleEntities(_)) => {
                        warn!(
                    "Insert global state command failed, multiple entities have the `GlobalStateMarker` component."
                );
                        return;
                    }
                }
            }
        };

        // Register storage for state `S`.
        let new_data = StateData::new(self.next);
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
            (Some(_), false) => {
                warn!(
                    "Attempted to insert state {}, but it was already present.",
                    type_name::<S>()
                );
            }
        }
    }
}

struct NextStateCommand<S: State> {
    local: Option<Entity>,
    next: Option<S>,
}

impl<S: State> NextStateCommand<S> {
    fn new(local: Option<Entity>, next: Option<S>) -> Self {
        Self { local, next }
    }
}

impl<S: State> Command for NextStateCommand<S> {
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
/// For [`Commands`] this will be a deferred operation, but for everything else the effect will be immediate.
pub trait StatesExt {
    /// Inserts a global state.
    /// If `overwrite` is enabled, this will override the existing state.
    /// If `local` is `None`, this will insert the global state.
    fn insert_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>, overwrite: bool);

    /// Set the next value of the state.
    /// This value will be used to update the state in the [`StateTransition`](crate::state::StateTransition) schedule.
    /// If `local` is `None`, this will update the global state.
    fn next_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>);

    /// Registers state in the world.
    fn register_state<S: State>(&mut self);
}

impl StatesExt for Commands<'_, '_> {
    fn insert_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>, overwrite: bool) {
        self.add(InsertStateCommand::new(local, next, overwrite))
    }

    fn next_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>) {
        self.add(NextStateCommand::new(local, next))
    }

    fn register_state<S: State>(&mut self) {
        self.add(|world: &mut World| {
            S::register_state(world);
        });
    }
}

impl StatesExt for World {
    fn insert_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>, overwrite: bool) {
        InsertStateCommand::new(local, next, overwrite).apply(self);
    }

    fn next_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>) {
        NextStateCommand::new(local, next).apply(self);
    }

    fn register_state<S: State>(&mut self) {
        S::register_state(self);
    }
}

#[cfg(feature = "bevy_app")]
impl StatesExt for bevy_app::SubApp {
    fn insert_state<S: State>(&mut self, local: Option<Entity>, value: Option<S>, overwrite: bool) {
        self.world_mut().insert_state(local, value, overwrite);
    }

    fn next_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>) {
        self.world_mut().next_state(local, next);
    }

    fn register_state<S: State>(&mut self) {
        self.world_mut().register_state::<S>();
    }
}

#[cfg(feature = "bevy_app")]
impl StatesExt for bevy_app::App {
    fn insert_state<S: State>(&mut self, local: Option<Entity>, value: Option<S>, overwrite: bool) {
        self.main_mut().insert_state(local, value, overwrite);
    }

    fn next_state<S: State>(&mut self, local: Option<Entity>, next: Option<S>) {
        self.main_mut().next_state(local, next);
    }

    fn register_state<S: State>(&mut self) {
        self.main_mut().register_state::<S>();
    }
}
