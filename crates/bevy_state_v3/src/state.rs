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
    data::{StateData, StateUpdateCurrent, StateUpdateDependency},
    events::{StateEnter, StateExit},
};

#[derive(Debug, PartialEq, Eq, Hash, Clone, ScheduleLabel)]
pub struct StateTransition;

#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
/// System set for ordering state machinery.
/// This consists of 3 steps:
/// - updates - check for state change requests, evaluate new values and propagate updates to children,
/// - exits - runs transitions from bottom to top,
/// - enters - runs transitions from top to bottom.
///
/// Each step is divided into smaller steps based on the state's order in hierarchy.
pub enum StateSystemSet {
    /// Group for all updates.
    AllUpdates,
    /// Update at a specific order, from bottom (root) to top (leaf).
    Update(u32),
    /// Group for all exits.
    AllExits,
    /// Exit at a specific order, from top (leaf) to bottom (root).
    Exit(u32),
    /// Group for all enters.
    AllEnters,
    /// Enter at a specific order, from bottom (root) to top (leaf).
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

/// Trait for types that act as a state.
pub trait State: Sized + Clone + Debug + PartialEq + Send + Sync + 'static {
    /// Parent states which this state depends on.
    type DependencySet: StateSet;

    /// Never set this to 0.
    const ORDER: u32 = Self::DependencySet::HIGHEST_ORDER + 1;

    /// Called when a [`StateData::next`] value is set or any of the [`Self::Dependencies`] change.
    /// If the returned value is [`Some`] it's used to update the [`StateData<Self>`].
    fn update<'a>(
        state: StateUpdateCurrent<Self>,
        dependencies: <<Self as State>::DependencySet as StateSet>::UpdateDependencies<'a>,
    ) -> Option<Option<Self>>;

    /// Registers this state in the world together with all dependencies.
    fn register_state(world: &mut World) {
        Self::DependencySet::register_states(world);

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
            state.updated = false;
            let is_dependency_set_changed = Self::DependencySet::is_changed(&dependencies);
            let is_target_changed = state.target.is_some();
            if is_dependency_set_changed || is_target_changed {
                if let Some(next) = Self::update(
                    (&*state).into(),
                    Self::DependencySet::as_state_update_dependency(dependencies),
                ) {
                    state.target.take();
                    state.update(next);
                }
            }
        }
    }

    fn exit_system(
        mut commands: Commands,
        query: Query<(Entity, &StateData<Self>, Has<GlobalStateMarker>)>,
    ) {
        for (entity, state, is_global) in query.iter() {
            if !state.updated {
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
            if !state.updated {
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
    type UpdateDependencies<'a>;

    const HIGHEST_ORDER: u32;

    /// Registers all elements as required components.
    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    );

    /// Registers all required states.
    fn register_states(world: &mut World);

    /// Check dependencies for changes.
    fn is_changed(set: &<Self::Query as WorldQuery>::Item<'_>) -> bool;

    fn as_state_update_dependency<'a>(
        set: <Self::Query as WorldQuery>::Item<'a>,
    ) -> Self::UpdateDependencies<'a>;
}

impl StateSet for () {
    type Query = ();
    type UpdateDependencies<'a> = ();

    const HIGHEST_ORDER: u32 = 0;

    fn register_required_components(
        _components: &mut Components,
        _storages: &mut Storages,
        _required_components: &mut RequiredComponents,
    ) {
    }

    fn register_states(_world: &mut World) {}

    fn is_changed(_set: &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        false
    }

    fn as_state_update_dependency<'a>(
        _set: <Self::Query as WorldQuery>::Item<'a>,
    ) -> Self::UpdateDependencies<'a> {
    }
}

impl<S1: State> StateSet for S1 {
    type Query = &'static StateData<S1>;
    type UpdateDependencies<'a> = StateUpdateDependency<'a, S1>;

    const HIGHEST_ORDER: u32 = S1::ORDER;

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
    }

    fn register_states(world: &mut World) {
        S1::register_state(world);
    }

    fn is_changed(s1: &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        s1.updated
    }

    fn as_state_update_dependency<'a>(
        s1: <Self::Query as WorldQuery>::Item<'a>,
    ) -> Self::UpdateDependencies<'a> {
        s1.into()
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
            type UpdateDependencies<'a> = ($(StateUpdateDependency<'a, $type>, )*);

            const HIGHEST_ORDER: u32 = max!($($type::ORDER), +);

            fn register_required_components(
                components: &mut Components,
                storages: &mut Storages,
                required_components: &mut RequiredComponents,
            ) {
                $(required_components.register(components, storages, StateData::<$type>::default);)
                +
            }

            fn register_states(world: &mut World) {
                $($type::register_state(world);)
                +
            }

            fn is_changed(($($var, )+): &<Self::Query as WorldQuery>::Item<'_>) -> bool {
                $($var.updated) || +
            }

            fn as_state_update_dependency<'a>(
                ($($var, )+): <Self::Query as WorldQuery>::Item<'a>,
            ) -> Self::UpdateDependencies<'a> {
                ($($var.into(), )+)
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
