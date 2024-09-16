use std::any::type_name;

use bevy_ecs::{
    entity::Entity,
    query::{QuerySingleError, With},
    system::Commands,
    world::{Command, World},
};
use bevy_utils::tracing::warn;

use crate::{
    data::StateData,
    state::{GlobalStateMarker, State, StateTransitionsConfig, StateUpdate},
};

struct InitializeStateCommand<S: State> {
    local: Option<Entity>,
    initial: Option<S>,
    suppress_initial_update: bool,
}

impl<S: State> InitializeStateCommand<S> {
    fn new(local: Option<Entity>, initial: Option<S>, suppress_initial_update: bool) -> Self {
        Self {
            local,
            initial,
            suppress_initial_update,
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
                world.entity_mut(entity).insert(StateData::<S>::new(
                    self.initial,
                    self.suppress_initial_update,
                ));
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

struct SetStateTargetCommand<S: State<Target = StateUpdate<S>>> {
    local: Option<Entity>,
    target: Option<S>,
}

impl<S: State<Target = StateUpdate<S>>> SetStateTargetCommand<S> {
    fn new(local: Option<Entity>, target: Option<S>) -> Self {
        Self { local, target }
    }
}

impl<S: State<Target = StateUpdate<S>>> Command for SetStateTargetCommand<S> {
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
        state.target = match self.target {
            Some(s) => StateUpdate::Enable(s),
            None => StateUpdate::Disable,
        };
    }
}

#[doc(hidden)]
/// All of the operations can happen immediatelly (with [`World`], [`SubApp`](bevy_app::SubApp), [`App`](bevy_app::App)) or in a deferred manner (with [`Commands`]).
pub trait StatesExt {
    /// Registers machinery for this state as well as all dependencies.
    fn register_state<S: State>(&mut self, config: StateTransitionsConfig<S>);

    /// Adds the state to the provided `local` entity or otherwise the global state.
    /// If initial update is suppresed, no initial transitions will be generated.
    /// The state added this way is always disabled and has to be enabled through [`next_state`] method.
    /// This also adds all dependencies through required components.
    fn init_state<S: State>(
        &mut self,
        local: Option<Entity>,
        initial: Option<S>,
        suppress_initial_update: bool,
    );

    /// Sets the [`State::Target`] value in [`StateData`],
    /// which will result in an [`State::update`] call during [`StateTransition`](crate::state::StateTransition) schedule.
    /// Much like [`StatesExt::init_state`] you need to provide a local entity or nothing, for global state.
    ///
    /// This only works with the [`StateUpdate`] target.
    fn state_target<S: State<Target = StateUpdate<S>>>(
        &mut self,
        local: Option<Entity>,
        target: Option<S>,
    );
}

impl StatesExt for Commands<'_, '_> {
    fn register_state<S: State>(&mut self, config: StateTransitionsConfig<S>) {
        self.add(|world: &mut World| {
            S::register_state(world, config, false);
        });
    }

    fn init_state<S: State>(
        &mut self,
        local: Option<Entity>,
        initial: Option<S>,
        suppress_initial_update: bool,
    ) {
        self.add(InitializeStateCommand::<S>::new(
            local,
            initial,
            suppress_initial_update,
        ))
    }

    fn state_target<S: State<Target = StateUpdate<S>>>(
        &mut self,
        local: Option<Entity>,
        target: Option<S>,
    ) {
        self.add(SetStateTargetCommand::new(local, target))
    }
}

impl StatesExt for World {
    fn register_state<S: State>(&mut self, config: StateTransitionsConfig<S>) {
        S::register_state(self, config, false);
    }

    fn init_state<S: State>(
        &mut self,
        local: Option<Entity>,
        initial: Option<S>,
        suppress_initial_update: bool,
    ) {
        InitializeStateCommand::<S>::new(local, initial, suppress_initial_update).apply(self);
    }

    fn state_target<S: State<Target = StateUpdate<S>>>(
        &mut self,
        local: Option<Entity>,
        target: Option<S>,
    ) {
        SetStateTargetCommand::new(local, target).apply(self);
    }
}

#[cfg(feature = "bevy_app")]
impl StatesExt for bevy_app::SubApp {
    fn register_state<S: State>(&mut self, config: StateTransitionsConfig<S>) {
        self.world_mut().register_state::<S>(config);
    }

    fn init_state<S: State>(
        &mut self,
        local: Option<Entity>,
        initial: Option<S>,
        suppress_initial_update: bool,
    ) {
        self.world_mut()
            .init_state::<S>(local, initial, suppress_initial_update);
    }

    fn state_target<S: State<Target = StateUpdate<S>>>(
        &mut self,
        local: Option<Entity>,
        target: Option<S>,
    ) {
        self.world_mut().state_target(local, target);
    }
}

#[cfg(feature = "bevy_app")]
impl StatesExt for bevy_app::App {
    fn register_state<S: State>(&mut self, config: StateTransitionsConfig<S>) {
        self.main_mut().register_state::<S>(config);
    }

    fn init_state<S: State>(
        &mut self,
        local: Option<Entity>,
        initial: Option<S>,
        suppress_initial_update: bool,
    ) {
        self.main_mut()
            .init_state::<S>(local, initial, suppress_initial_update);
    }

    fn state_target<S: State<Target = StateUpdate<S>>>(
        &mut self,
        local: Option<Entity>,
        target: Option<S>,
    ) {
        self.main_mut().state_target(local, target);
    }
}
