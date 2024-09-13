use std::any::type_name;

use bevy_ecs::{
    component::{Components, RequiredComponents},
    entity::Entity,
    observer::{Observer, Trigger},
    query::{QuerySingleError, ReadOnlyQueryData, With, WorldQuery},
    schedule::{ScheduleLabel, Schedules},
    storage::Storages,
    system::{Commands, Query},
    world::World,
};
use bevy_utils::tracing::{info, warn};

use crate::{data::StateData, events::StateUpdate, StateEdge};

#[derive(Debug, PartialEq, Eq, Hash, Clone, ScheduleLabel)]
pub struct StateTransition;

/// Trait for types that act as a state.
pub trait State: Sized + PartialEq + Send + Sync + 'static {
    /// Parent states which this state depends on.
    type Dependencies: StateSet;

    /// Order in the state update hierarchy.
    const ORDER: usize = Self::Dependencies::ORDER + 1;

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
            Observer::new(
                |trigger: Trigger<StateUpdate<Self>>,
                 mut state: Query<&mut StateData<Self>>,
                 dependencies: Query<<Self::Dependencies as StateSet>::Data>| {
                    let entity = trigger.entity();
                    let mut state = state.get_mut(entity).unwrap();
                    let dependencies = dependencies.get(entity).unwrap();
                    let next = state.next.take();
                    if let Some(next) = Self::update(next, dependencies) {
                        state.advance(next);
                    }
                    info!("Update on: {:?}", type_name::<Self>());
                },
            ),
        ));

        // Register propagation from parent states.
        Self::Dependencies::register_update_propagation::<Self>(world);

        let mut schedules = world.resource_mut::<Schedules>();
        schedules.entry(StateTransition).add_systems(
            |states: Query<(Entity, &StateData<Self>)>, mut commands: Commands| {
                for (entity, state) in states.iter() {
                    if state.next().is_some() {
                        commands.trigger_targets(StateUpdate::<Self>::default(), entity);
                    }
                }
            },
        );
    }
}

/// All possible combinations of state dependencies.
pub trait StateSet {
    /// Highest order of source states.
    const ORDER: usize;

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

fn register_propatation<P: State, C: State>(world: &mut World) {
    world.spawn((
        StateEdge::<P, C>::default(),
        Observer::new(|trigger: Trigger<StateUpdate<P>>, mut commands: Commands| {
            let entity = trigger.entity();
            commands.trigger_targets(StateUpdate::<C>::default(), entity);
        }),
    ));
}

impl StateSet for () {
    const ORDER: usize = 0;
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
    const ORDER: usize = S1::ORDER;
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
impl<S1: State, S2: State> StateSet for (S1, S2) {
    const ORDER: usize = usize_max(S1::ORDER, S2::ORDER);
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
    const ORDER: usize = usize_max(S1::ORDER, usize_max(S2::ORDER, S3::ORDER));
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
/// Const time usize max.
const fn usize_max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
