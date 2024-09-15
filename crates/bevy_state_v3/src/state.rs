use std::{any::type_name, fmt::Debug, marker::PhantomData, u32};

use bevy_ecs::{
    component::{Component, Components, RequiredComponents},
    entity::Entity,
    query::{Has, QuerySingleError, ReadOnlyQueryData, With, WorldQuery},
    schedule::{IntoSystemConfigs, IntoSystemSetConfigs, ScheduleLabel, Schedules, SystemSet},
    storage::Storages,
    system::{Commands, Query},
    world::World,
};
use bevy_utils::{all_tuples, tracing::warn};

use crate::{
    data::StateData,
    events::{StateEnter, StateExit},
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
    fn register_state(world: &mut World) {
        Self::DependencySet::register_required_states(world);

        match world
            .query_filtered::<(), With<RegisteredState<Self>>>()
            .get_single(world)
        {
            // Already registered, skip.
            Ok(_) => {
                return;
            }
            Err(QuerySingleError::MultipleEntities(_)) => {
                warn!(
                    "Failed to register state {}, edge already registered multiple times.",
                    type_name::<Self>()
                );
                return;
            }
            // Not registered, continue.
            Err(QuerySingleError::NoEntities(_)) => {}
        }

        world.spawn(RegisteredState::<Self>::default());

        // Register systems for this state.
        let mut schedules = world.resource_mut::<Schedules>();
        let schedule = schedules.entry(StateTransition);
        schedule.configure_sets(StateSystemSet::configuration::<Self>());
        schedule.add_systems(Self::update_system.in_set(StateSystemSet::update::<Self>()));
        schedule.add_systems(Self::exit_system.in_set(StateSystemSet::exit::<Self>()));
        schedule.add_systems(Self::enter_system.in_set(StateSystemSet::enter::<Self>()));
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
            let is_target_changed = state.target.is_changed();
            if is_dependency_set_changed || is_target_changed {
                let result = Self::update(&mut state, dependencies);
                if let Some(next) = result.as_options() {
                    state.update(next);
                    state.target.reset();
                }
            }
        }
    }

    fn exit_system(
        mut commands: Commands,
        query: Query<(Entity, &StateData<Self>, Has<GlobalStateMarker>)>,
    ) {
        for (entity, state, is_global) in query.iter() {
            if !state.is_updated {
                continue;
            }
            let target = is_global.then_some(Entity::PLACEHOLDER).unwrap_or(entity);
            commands.trigger_targets(StateExit::<Self>::default(), target);
        }
    }

    fn enter_system(
        mut commands: Commands,
        states: Query<(Entity, &StateData<Self>, Has<GlobalStateMarker>)>,
    ) {
        for (entity, state, is_global) in states.iter() {
            if !state.is_updated {
                continue;
            }
            let target = is_global.then_some(Entity::PLACEHOLDER).unwrap_or(entity);
            commands.trigger_targets(StateEnter::<Self>::default(), target);
        }
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
        S1::register_state(world);
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
                $($type::register_state(world);)
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

impl<S: State> StateTarget for StateUpdate<S> {
    fn is_changed(&self) -> bool {
        self.is_something()
    }

    fn reset(&mut self) {
        self.take();
    }
}

/// Variable target backend for states.
/// Different backends can allow for different features:
/// - singular requests,
/// - mutable state that tracks changes,
/// - stack of states.
pub trait StateTarget: Default + Send + Sync + 'static {
    /// Returns whether state should be updated.
    fn is_changed(&self) -> bool;

    /// Resets the target, usually by disabling the [`Self::is_changed`] until another request.
    fn reset(&mut self);
}
