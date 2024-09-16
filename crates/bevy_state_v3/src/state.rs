use std::{any::type_name, fmt::Debug, marker::PhantomData, u32};

use bevy_ecs::{
    component::{Component, Components, RequiredComponents},
    entity::Entity,
    query::{Has, QuerySingleError, ReadOnlyQueryData, With, WorldQuery},
    schedule::{
        IntoSystemConfigs, IntoSystemSetConfigs, ScheduleLabel, Schedules, SystemConfigs, SystemSet,
    },
    storage::Storages,
    system::{Commands, Query},
    world::World,
};
use bevy_utils::{all_tuples, tracing::warn};

use crate::{
    data::StateData,
    events::{OnEnter, OnExit},
};

#[derive(Debug, PartialEq, Eq, Hash, Clone, ScheduleLabel)]
pub struct StateTransition;

#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
/// The `StateTransition` schedule runs 3 system sets:
/// - [`AllUpdates`] - Updates based on `target` and dependency changes from root states to leaf states, sets the `updated` flag.
/// - [`AllExits`] - Triggers [`StateExit<S>`] observers from leaf states to root states, targeted for local state, untargeted for global state.
/// - [`AllEnters`] - Triggers [`StateEnter<S>`] observers from root states to leaf states, targeted for local state, untargeted for global state.
/// Smaller sets are used to specify order in the grap.
/// Order is derived when specifying state dependencies, smaller value meaning closer to root.
pub enum StateSystemSet {
    /// All [`Update`]s.
    AllUpdates,
    /// Lower values before higher ones.
    Update(u32),
    /// All [`Exit`]s.
    AllExits,
    /// Higher values then lower ones.
    Exit(u32),
    /// All [`Enter`]s.
    AllEnters,
    /// Same as [`Update`], lower values before higher ones.
    Enter(u32),
}

impl StateSystemSet {
    pub fn update<S: State>() -> Self {
        Self::Update(S::ORDER)
    }

    pub fn exit<S: State>() -> Self {
        Self::Exit(S::ORDER)
    }

    pub fn enter<S: State>() -> Self {
        Self::Enter(S::ORDER)
    }

    pub fn configuration<S: State>() -> impl IntoSystemSetConfigs {
        (
            (Self::AllUpdates, Self::AllExits, Self::AllEnters).chain(),
            Self::update::<S>()
                .after(Self::Update(S::ORDER - 1))
                .in_set(Self::AllUpdates),
            Self::exit::<S>()
                .before(Self::Exit(S::ORDER - 1))
                .in_set(Self::AllExits),
            Self::enter::<S>()
                .after(Self::Enter(S::ORDER - 1))
                .in_set(Self::AllEnters),
        )
    }
}

pub type StateDependencies<'a, S> =
    <<<S as State>::DependencySet as StateSet>::Query as WorldQuery>::Item<'a>;

pub struct StateTransitionsConfig<S: State> {
    systems: Vec<SystemConfigs>,
    _state: PhantomData<S>,
}

impl<S: State> Default for StateTransitionsConfig<S> {
    fn default() -> Self {
        Self {
            systems: vec![
                on_exit_transition::<S>
                    .in_set(StateSystemSet::exit::<S>())
                    .into(),
                on_enter_transition::<S>
                    .in_set(StateSystemSet::enter::<S>())
                    .into(),
            ],
            _state: Default::default(),
        }
    }
}

impl<S: State> StateTransitionsConfig<S> {
    /// Config that creates no transitions.
    /// For standard [`OnExit`] and [`OnEnter`] use the [`StateTransitionsConfig::default`].
    pub fn empty() -> Self {
        Self {
            systems: vec![],
            _state: PhantomData,
        }
    }

    /// Adds a system to run when state is exited.
    /// An example system that runs [`OnExit`] is [`on_exit_transition`].
    pub fn with_on_exit<M>(mut self, system: impl IntoSystemConfigs<M>) -> Self {
        self.systems
            .push(system.in_set(StateSystemSet::exit::<S>()));
        self
    }

    /// Adds a system to run when state is entered.
    /// An example system that runs [`OnEnter`] is [`on_enter_transition`].
    pub fn with_on_enter<M>(mut self, system: impl IntoSystemConfigs<M>) -> Self {
        self.systems
            .push(system.in_set(StateSystemSet::enter::<S>()));
        self
    }
}

/// Trait for types that act as a state.
pub trait State: Sized + Clone + Debug + PartialEq + Send + Sync + 'static {
    /// Parent states which this state depends on.
    type DependencySet: StateSet;

    /// Backing structure for picking state target.
    type Target: StateTarget;

    /// Never set this to 0.
    const ORDER: u32 = Self::DependencySet::HIGHEST_ORDER + 1;

    /// Called when a [`StateData::next`] value is set or any of the [`Self::Dependencies`] change.
    /// If the returned value is [`Some`] it's used to update the [`StateData<Self>`].
    fn update(
        state: &mut StateData<Self>,
        dependencies: StateDependencies<'_, Self>,
    ) -> StateUpdate<Self>;

    /// Registers this state in the world together with all dependencies.
    fn register_state(
        world: &mut World,
        transitions: StateTransitionsConfig<Self>,
        recursive: bool,
    ) {
        Self::DependencySet::register_required_states(world);

        match world
            .query_filtered::<(), With<RegisteredState<Self>>>()
            .get_single(world)
        {
            Ok(_) => {
                // Skip warnings from recursive registers.
                if !recursive {
                    warn!(
                        "State {} is already registered, additional configuration will be ignored.",
                        type_name::<Self>()
                    );
                }
                return;
            }
            Err(QuerySingleError::MultipleEntities(_)) => {
                warn!(
                    "Failed to register state {}, edge already registered multiple times.",
                    type_name::<Self>()
                );
                return;
            }
            Err(QuerySingleError::NoEntities(_)) => {}
        }

        world.spawn(RegisteredState::<Self>::default());

        // Register systems for this state.
        let mut schedules = world.resource_mut::<Schedules>();
        let schedule = schedules.entry(StateTransition);
        schedule.configure_sets(StateSystemSet::configuration::<Self>());
        schedule.add_systems(Self::update_system.in_set(StateSystemSet::update::<Self>()));
        for system in transitions.systems {
            schedule.add_systems(system);
        }
    }

    fn update_system(
        mut query: Query<(
            &mut StateData<Self>,
            <Self::DependencySet as StateSet>::Query,
        )>,
    ) {
        for (mut state, dependencies) in query.iter_mut() {
            state.is_updated = false;
            let is_dependency_set_changed = Self::DependencySet::is_changed(&dependencies);
            let is_target_changed = state.target.should_update();
            if is_dependency_set_changed || is_target_changed {
                let result = Self::update(&mut state, dependencies);
                if let Some(next) = result.as_options() {
                    state.update(next);
                    state.target.reset();
                }
            }
        }
    }
}

pub fn on_exit_transition<S: State>(
    mut commands: Commands,
    query: Query<(Entity, &StateData<S>, Has<GlobalStateMarker>)>,
) {
    for (entity, state, is_global) in query.iter() {
        if !state.is_updated || state.is_reentrant() {
            continue;
        }
        let target = is_global.then_some(Entity::PLACEHOLDER).unwrap_or(entity);
        commands.trigger_targets(
            OnExit::<S>::new(state.previous().cloned(), state.current().cloned()),
            target,
        );
    }
}

pub fn on_enter_transition<S: State>(
    mut commands: Commands,
    states: Query<(Entity, &StateData<S>, Has<GlobalStateMarker>)>,
) {
    for (entity, state, is_global) in states.iter() {
        if !state.is_updated || state.is_reentrant() {
            continue;
        }
        let target = is_global.then_some(Entity::PLACEHOLDER).unwrap_or(entity);
        commands.trigger_targets(
            OnEnter::<S>::new(state.previous().cloned(), state.current().cloned()),
            target,
        );
    }
}

/// All possible combinations of state dependencies.
pub trait StateSet {
    /// Parameters provided to [`State::on_update`].
    type Query: ReadOnlyQueryData;

    const HIGHEST_ORDER: u32;

    /// Registers all elements as required components.
    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    );

    /// Registers all required states.
    fn register_required_states(world: &mut World);

    /// Check dependencies for changes.
    fn is_changed(set: &<Self::Query as WorldQuery>::Item<'_>) -> bool;
}

impl StateSet for () {
    type Query = ();

    const HIGHEST_ORDER: u32 = 0;

    fn register_required_components(
        _components: &mut Components,
        _storages: &mut Storages,
        _required_components: &mut RequiredComponents,
    ) {
    }

    fn register_required_states(_world: &mut World) {}

    fn is_changed(_set: &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        false
    }
}

impl<S1: State> StateSet for S1 {
    type Query = &'static StateData<S1>;

    const HIGHEST_ORDER: u32 = S1::ORDER;

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
    }

    fn register_required_states(world: &mut World) {
        S1::register_state(world, StateTransitionsConfig::default(), true);
    }

    fn is_changed(s1: &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        s1.is_updated
    }
}

const fn const_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

macro_rules! max {
    ($a:expr) => ( $a );
    ($a:expr, $b:expr) => {
        const_max($a, $b)
    };
    ($a:expr, $b:expr, $($other:expr), *) => {
        max!(const_max($a, $b), $($other), +)
    };
}

macro_rules! impl_state_set {
    ($(#[$meta:meta])* $(($type:ident, $var:ident)), *) => {
        $(#[$meta])*
        impl<$($type: State), *> StateSet for ($($type, )*) {
            type Query = ($(&'static StateData<$type>, )*);

            const HIGHEST_ORDER: u32 = max!($($type::ORDER), +);

            fn register_required_components(
                components: &mut Components,
                storages: &mut Storages,
                required_components: &mut RequiredComponents,
            ) {
                $(required_components.register(components, storages, StateData::<$type>::default);)
                +
            }

            fn register_required_states(world: &mut World) {
                $($type::register_state(world, StateTransitionsConfig::default(), true);)
                +
            }

            fn is_changed(($($var, )+): &<Self::Query as WorldQuery>::Item<'_>) -> bool {
                $($var.is_updated) || +
            }
        }
    };
}

all_tuples!(
    #[doc(fake_variadic)]
    impl_state_set,
    1,
    15,
    S,
    s
);

/// Marker component for global states.
#[derive(Component)]
pub struct GlobalStateMarker;

/// Used to keep track of which states are registered and which aren't.
#[derive(Component)]
pub struct RegisteredState<S: State>(PhantomData<S>);

impl<S: State> Default for RegisteredState<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Default, Debug, Clone)]
pub enum StateUpdate<S> {
    #[default]
    Nothing,
    Disable,
    Enable(S),
}

impl<S> StateUpdate<S> {
    pub fn is_something(&self) -> bool {
        if let Self::Nothing = self {
            false
        } else {
            true
        }
    }

    pub fn as_options(self) -> Option<Option<S>> {
        match self {
            StateUpdate::Nothing => None,
            StateUpdate::Disable => Some(None),
            StateUpdate::Enable(s) => Some(Some(s)),
        }
    }

    pub fn as_ref(&self) -> StateUpdate<&S> {
        match &self {
            StateUpdate::Nothing => StateUpdate::Nothing,
            StateUpdate::Disable => StateUpdate::Disable,
            StateUpdate::Enable(s) => StateUpdate::Enable(s),
        }
    }

    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

/// Variable target backend for states.
/// Different backends can allow for different features:
/// - [`()`] for no manual updates, only dependency based ones (computed states),
/// - [`StateUpdate`] for overwrite-style control (root/sub states),
/// - mutable target state, for combining multiple requests,
/// - stack or vector of states.
pub trait StateTarget: Default + Send + Sync + 'static {
    /// Returns whether the state should be updated.
    fn should_update(&self) -> bool;

    /// Resets the target to reset change detection.
    fn reset(&mut self);
}

impl<S: State> StateTarget for StateUpdate<S> {
    fn should_update(&self) -> bool {
        self.is_something()
    }

    fn reset(&mut self) {
        self.take();
    }
}

impl StateTarget for () {
    fn should_update(&self) -> bool {
        false
    }

    fn reset(&mut self) {}
}
