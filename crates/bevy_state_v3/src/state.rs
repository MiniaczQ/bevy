use std::{any::type_name, fmt::Debug, u32};

use bevy_ecs::{
    query::{QuerySingleError, With},
    schedule::{IntoSystemConfigs, Schedules},
    system::Query,
    world::World,
};
use bevy_utils::tracing::warn;

use crate::{
    data::{RegisteredState, StateData, StateTarget, StateUpdate}, scheduling::{StateSystemSet, StateTransition}, state_set::{StateDependencies, StateSet}, transitions::StateTransitionsConfig
};

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
