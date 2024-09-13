use std::{any::type_name, marker::PhantomData};

use bevy_ecs::{
    component::{Component, Components, RequiredComponents},
    entity::Entity,
    observer::{Observer, Trigger},
    query::{QuerySingleError, ReadOnlyQueryData, With, WorldQuery},
    schedule::{ScheduleLabel, Schedules},
    storage::Storages,
    system::{Commands, Query},
    world::World,
};
use bevy_utils::tracing::warn;

use crate::{
    data::StateData,
    events::{OnTransition, OnUpdate},
};

#[derive(Debug, PartialEq, Eq, Hash, Clone, ScheduleLabel)]
pub struct StateTransition;

/// Trait for types that act as a state.
pub trait State: Sized + PartialEq + Send + Sync + 'static {
    /// Parent states which this state depends on.
    type Dependencies: StateSet;

    /// Called when a [`StateData::next`] value is set or any of the [`Self::Dependencies`] change.
    /// If the returned value is [`Some`] it's used to update the [`StateData<Self>`].
    fn update(
        next: Option<Option<Self>>,
        dependencies: <<<Self as State>::Dependencies as StateSet>::Data as WorldQuery>::Item<'_>,
    ) -> Option<Option<Self>>;

    /// Registers the state in the world.
    fn register_state(world: &mut World) {
        match world
            .query_filtered::<(), With<StateEdge<Self, Self>>>()
            .get_single(world)
        {
            Err(QuerySingleError::MultipleEntities(_)) => {
                warn!(
                    "Failed to register state {}, edge already registered multiple times.",
                    type_name::<Self>()
                );
                return;
            }
            Ok(_) => {
                warn!("State {} already registered.", type_name::<Self>());
                return;
            }
            Err(QuerySingleError::NoEntities(_)) => {}
        }

        // Register update observer for state `S`.
        world.spawn((
            StateEdge::<Self, Self>::default(),
            Observer::new(state_update::<Self>),
        ));

        // Register propagation from parent states.
        Self::Dependencies::register_update_propagation::<Self>(world);

        let mut schedules = world.resource_mut::<Schedules>();
        schedules.entry(StateTransition).add_systems(
            |states: Query<(Entity, &StateData<Self>)>, mut commands: Commands| {
                for (entity, state) in states.iter() {
                    if state.next().is_some() {
                        commands.trigger_targets(OnUpdate::<Self>::default(), entity);
                    }
                }
            },
        );
    }
}

fn state_update<S: State>(
    trigger: Trigger<OnUpdate<S>>,
    mut state: Query<&mut StateData<S>>,
    dependencies: Query<<S::Dependencies as StateSet>::Data>,
    mut commands: Commands,
) {
    let entity = trigger.entity();
    let mut state = state.get_mut(entity).unwrap();
    let dependencies = dependencies.get(entity).unwrap();
    let next = state.next.take();
    if let Some(next) = S::update(next, dependencies) {
        state.advance(next);
        commands.trigger_targets(OnTransition::<S>::default(), entity);
    }
}

/// All possible combinations of state dependencies.
pub trait StateSet {
    /// Parameters provided to [`State::on_update`].
    type Data: ReadOnlyQueryData;

    /// Registers all elements as required components.
    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    );

    /// Registers observers for update propagation.
    fn register_update_propagation<S: State>(world: &mut World);
}

impl StateSet for () {
    type Data = ();

    fn register_required_components(
        _components: &mut Components,
        _storages: &mut Storages,
        _required_components: &mut RequiredComponents,
    ) {
    }

    fn register_update_propagation<S: State>(_world: &mut World) {}
}

impl<S1: State> StateSet for S1 {
    type Data = &'static StateData<S1>;

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
    }

    fn register_update_propagation<S: State>(world: &mut World) {
        register_propatation::<S1, S>(world);
    }
}

// TODO: use `all_tuples_with_size!()``
impl<S1: State, S2: State> StateSet for (S1, S2) {
    type Data = (&'static StateData<S1>, &'static StateData<S2>);

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
        required_components.register(components, storages, StateData::<S2>::default);
    }

    fn register_update_propagation<S: State>(world: &mut World) {
        register_propatation::<S1, S>(world);
        register_propatation::<S2, S>(world);
    }
}

impl<S1: State, S2: State, S3: State> StateSet for (S1, S2, S3) {
    type Data = (
        &'static StateData<S1>,
        &'static StateData<S2>,
        &'static StateData<S3>,
    );

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
        required_components.register(components, storages, StateData::<S2>::default);
        required_components.register(components, storages, StateData::<S3>::default);
    }

    fn register_update_propagation<S: State>(world: &mut World) {
        register_propatation::<S1, S>(world);
        register_propatation::<S2, S>(world);
        register_propatation::<S3, S>(world);
    }
}

fn register_propatation<P: State, C: State>(world: &mut World) {
    world.spawn((
        StateEdge::<P, C>::default(),
        Observer::new(
            |trigger: Trigger<OnTransition<P>>, mut commands: Commands| {
                let entity = trigger.entity();
                commands.trigger_targets(OnUpdate::<C>::default(), entity);
            },
        ),
    ));
}

/// Marker component for global states.
#[derive(Component)]
pub struct GlobalStateMarker;

/// Edge between two states in a hierarchy.
/// `C` is dependent on `P`, all updates to `P` result in updates to `C`.
/// Edges between a type `S` with itself are used for running updates.
#[derive(Component)]
pub struct StateEdge<P: State, C: State>(PhantomData<(P, C)>);

impl<P: State, C: State> Default for StateEdge<P, C> {
    fn default() -> Self {
        Self(PhantomData::default())
    }
}
