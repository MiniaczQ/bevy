use std::{any::type_name, marker::PhantomData};

use bevy_ecs::{
    entity::Entity,
    query::{QuerySingleError, With},
    system::Commands,
    world::{Command, World},
};
use bevy_utils::tracing::warn;

use crate::{
    data::StateData,
    state::{GlobalStateMarker, State},
};

struct InitializeStateCommand<S: State> {
    local: Option<Entity>,
    suppress_initial_update: bool,
    _data: PhantomData<S>,
}

impl<S: State> InitializeStateCommand<S> {
    fn new(local: Option<Entity>, suppress_initial_update: bool) -> Self {
        Self {
            local,
            suppress_initial_update,
            _data: PhantomData::default(),
        }
    }
}

impl<S: State + Send + Sync + 'static> Command for InitializeStateCommand<S> {
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
        let state_data = world
            .query::<&mut StateData<S>>()
            .get_mut(world, entity)
            .ok();
        match state_data {
            None => {
                world
                    .entity_mut(entity)
                    .insert(StateData::<S>::new(self.suppress_initial_update));
            }
            Some(_) => {
                warn!(
                    "Attempted to initialize state {}, but it was already present.",
                    type_name::<S>()
                );
            }
        }
    }
}

struct SetStateTargetCommand<S: State> {
    local: Option<Entity>,
    target: Option<S>,
}

impl<S: State> SetStateTargetCommand<S> {
    fn new(local: Option<Entity>, target: Option<S>) -> Self {
        Self { local, target }
    }
}

impl<S: State> Command for SetStateTargetCommand<S> {
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
        state.target = Some(self.target);
    }
}

#[doc(hidden)]
/// For [`Commands`] this will be a deferred operation, but for everything else the effect will be immediate.
pub trait StatesExt {
    /// Registers machinery for state.
    fn register_state<S: State>(&mut self);

    /// Initializes state.
    /// If `local` is `None`, this will work on the global state.
    fn init_state<S: State>(&mut self, local: Option<Entity>, suppress_initial_update: bool);

    /// Set the next value of the state.
    /// This value will be used to update the state in the [`StateTransition`](crate::state::StateTransition) schedule.
    /// If `local` is `None`, this will work on the global state.
    fn state_target<S: State>(&mut self, local: Option<Entity>, target: Option<S>);
}

impl StatesExt for Commands<'_, '_> {
    fn register_state<S: State>(&mut self) {
        self.add(|world: &mut World| {
            S::register_state(world);
        });
    }

    fn init_state<S: State>(&mut self, local: Option<Entity>, suppress_initial_update: bool) {
        self.add(InitializeStateCommand::<S>::new(
            local,
            suppress_initial_update,
        ))
    }

    fn state_target<S: State>(&mut self, local: Option<Entity>, target: Option<S>) {
        self.add(SetStateTargetCommand::new(local, target))
    }
}

impl StatesExt for World {
    fn register_state<S: State>(&mut self) {
        S::register_state(self);
    }

    fn init_state<S: State>(&mut self, local: Option<Entity>, suppress_initial_update: bool) {
        InitializeStateCommand::<S>::new(local, suppress_initial_update).apply(self);
    }

    fn state_target<S: State>(&mut self, local: Option<Entity>, target: Option<S>) {
        SetStateTargetCommand::new(local, target).apply(self);
    }
}

#[cfg(feature = "bevy_app")]
impl StatesExt for bevy_app::SubApp {
    fn register_state<S: State>(&mut self) {
        self.world_mut().register_state::<S>();
    }

    fn init_state<S: State>(&mut self, local: Option<Entity>, suppress_initial_update: bool) {
        self.world_mut()
            .init_state::<S>(local, suppress_initial_update);
    }

    fn state_target<S: State>(&mut self, local: Option<Entity>, target: Option<S>) {
        self.world_mut().state_target(local, target);
    }
}

#[cfg(feature = "bevy_app")]
impl StatesExt for bevy_app::App {
    fn register_state<S: State>(&mut self) {
        self.main_mut().register_state::<S>();
    }

    fn init_state<S: State>(&mut self, local: Option<Entity>, suppress_initial_update: bool) {
        self.main_mut()
            .init_state::<S>(local, suppress_initial_update);
    }

    fn state_target<S: State>(&mut self, local: Option<Entity>, target: Option<S>) {
        self.main_mut().state_target(local, target);
    }
}
