use std::{any::type_name, fmt::Debug, marker::PhantomData, u32};

use bevy_ecs::{
    component::{Component, Components, RequiredComponents},
    entity::Entity,
    query::{Has, QuerySingleError, ReadOnlyQueryData, With, WorldQuery},
    schedule::{IntoSystemConfigs, IntoSystemSetConfigs, ScheduleLabel, Schedules, SystemSet},
    storage::Storages,
    system::Query,
    world::World,
};
use bevy_utils::tracing::warn;

use crate::data::{StateData, StateUpdateCurrent, StateUpdateDependency};

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
            .query_filtered::<(), With<StateGraphNode<Self>>>()
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

        world.spawn(StateGraphNode::<Self>::default());

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

    fn exit_system(query: Query<(Entity, &StateData<Self>, Has<GlobalStateMarker>)>) {
        for (entity, state, is_global) in query.iter() {
            if !state.updated {
                continue;
            }
            // TODO: replace with transitions
            let pre = if is_global {
                "global".to_owned()
            } else {
                format!("{:?}", entity)
            };
            println!(
                "exit {} {} {:?} -> {:?}",
                pre,
                type_name::<Self>().split("::").last().unwrap(),
                state.previous(),
                state.current()
            );
        }
    }

    fn enter_system(query: Query<(Entity, &StateData<Self>, Has<GlobalStateMarker>)>) {
        for (entity, state, is_global) in query.iter() {
            if !state.updated {
                continue;
            }
            // TODO: replace with transitions
            let pre = if is_global {
                "global".to_owned()
            } else {
                format!("{:?}", entity)
            };
            println!(
                "enter {} {} {:?} -> {:?}",
                pre,
                type_name::<Self>().split("::").last().unwrap(),
                state.previous(),
                state.current()
            );
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

// TODO: use `all_tuples!()`
impl<S1: State, S2: State> StateSet for (S1, S2) {
    type Query = (&'static StateData<S1>, &'static StateData<S2>);
    type UpdateDependencies<'a> = (StateUpdateDependency<'a, S1>, StateUpdateDependency<'a, S2>);

    const HIGHEST_ORDER: u32 = max_u32(S1::ORDER, S2::ORDER);

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
        required_components.register(components, storages, StateData::<S2>::default);
    }

    fn register_states(world: &mut World) {
        S1::register_state(world);
        S2::register_state(world);
    }

    fn is_changed((s1, s2): &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        s1.updated || s2.updated
    }

    fn as_state_update_dependency<'a>(
        (s1, s2): <Self::Query as WorldQuery>::Item<'a>,
    ) -> Self::UpdateDependencies<'a> {
        (s1.into(), s2.into())
    }
}

impl<S1: State, S2: State, S3: State> StateSet for (S1, S2, S3) {
    type Query = (
        &'static StateData<S1>,
        &'static StateData<S2>,
        &'static StateData<S3>,
    );
    type UpdateDependencies<'a> = (
        StateUpdateDependency<'a, S1>,
        StateUpdateDependency<'a, S2>,
        StateUpdateDependency<'a, S3>,
    );

    const HIGHEST_ORDER: u32 = max_u32(max_u32(S1::ORDER, S2::ORDER), S3::ORDER);

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
        required_components.register(components, storages, StateData::<S2>::default);
        required_components.register(components, storages, StateData::<S3>::default);
    }

    fn register_states(world: &mut World) {
        S1::register_state(world);
        S2::register_state(world);
        S3::register_state(world);
    }

    fn is_changed((s1, s2, s3): &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        s1.updated || s2.updated || s3.updated
    }

    fn as_state_update_dependency<'a>(
        (s1, s2, s3): <Self::Query as WorldQuery>::Item<'a>,
    ) -> Self::UpdateDependencies<'a> {
        (s1.into(), s2.into(), s3.into())
    }
}

const fn max_u32(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Marker component for global states.
#[derive(Component)]
pub struct GlobalStateMarker;

/// Edge between two states in a hierarchy.
#[derive(Component)]
pub struct StateGraphEdge<Parent: State, Child: State>(PhantomData<(Parent, Child)>);

impl<Parent: State, Child: State> Default for StateGraphEdge<Parent, Child> {
    fn default() -> Self {
        Self(PhantomData::default())
    }
}

/// Node of a state.
#[derive(Component)]
pub struct StateGraphNode<S: State>(PhantomData<S>);

impl<S: State> Default for StateGraphNode<S> {
    fn default() -> Self {
        Self(PhantomData::default())
    }
}
