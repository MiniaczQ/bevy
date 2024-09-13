//! New states wahoo

mod commands;
mod data;
mod events;
mod state;

use std::marker::PhantomData;

use bevy_ecs::component::Component;
use state::State;

/// Marker component for global states.
#[derive(Component)]
pub struct GlobalStateMarker;

/// Edge between two states in a hierarchy.
/// `C` is dependent on `P`, all updates to `P` result in updates to `C`.
/// Edges between a type `S` with itself are used for running updates.
#[derive(Component)]
struct StateEdge<P: State, C: State>(PhantomData<(P, C)>);

impl<P: State, C: State> Default for StateEdge<P, C> {
    fn default() -> Self {
        Self(PhantomData::default())
    }
}

#[cfg(test)]
mod tests {
    use bevy_ecs::{query::WorldQuery, schedule::Schedules, world::World};

    use crate::{
        commands::CommandsExtStates,
        data::StateData,
        state::{StateSet, StateTransition},
        State,
    };

    #[derive(Debug, Default, PartialEq)]
    enum ManualState {
        #[default]
        A,
        B,
        C,
    }

    impl State for ManualState {
        type Dependencies = ();

        fn update(next: Option<Option<Self>>, _: Self::Dependencies) -> Option<Option<Self>> {
            // Pure manual control.
            // We ignore the update call from dependencies, because there are none.
            next
        }
    }

    #[derive(Debug, PartialEq)]
    struct ComputedState;

    impl State for ComputedState {
        type Dependencies = ManualState;

        fn update(
            next: Option<Option<Self>>,
            manual: <<<Self as State>::Dependencies as StateSet>::Data as WorldQuery>::Item<'_>,
        ) -> Option<Option<Self>> {
            match (manual.current(), next) {
                // If next was requested, ignore it.
                (_, Some(_)) => None,
                // If parent is valid, enable the state.
                (Some(ManualState::B), _) => Some(Some(ComputedState)),
                // If parent is invalid, disable the state.
                _ => Some(None),
            }
        }
    }

    #[derive(Debug, Default, PartialEq)]
    enum SubState {
        #[default]
        X,
        Y,
    }

    impl State for SubState {
        type Dependencies = ManualState;

        fn update(
            next: Option<Option<Self>>,
            manual: <<<Self as State>::Dependencies as StateSet>::Data as WorldQuery>::Item<'_>,
        ) -> Option<Option<Self>> {
            match (manual.current(), next) {
                // If parent state is valid, respect requested enabled next state.
                (Some(ManualState::C), Some(Some(next))) => Some(Some(next)),
                // If parent state is valid and requested next state disables the state, ignore it.
                (Some(ManualState::C), Some(None)) => None,
                // If parent state is valid and there was no next request, enable the state with default value.
                (Some(ManualState::C), None) => Some(Some(SubState::default())),
                // If parent state is invalid, disable the state.
                _ => Some(None),
            }
        }
    }

    fn get_global_state<S: State>(world: &mut World) -> &StateData<S> {
        world.query::<&StateData<S>>().single(world)
    }

    fn assert_current_state<S: State>(world: &mut World, _target: Option<S>) {
        let state = get_global_state::<S>(world);
        assert!(matches!(state.current(), _target));
    }

    #[test]
    fn test() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        ManualState::register_state(&mut world);
        ComputedState::register_state(&mut world);
        SubState::register_state(&mut world);

        world.insert_global_state(None::<ManualState>, false);
        world.insert_global_state(None::<ComputedState>, false);
        world.insert_global_state(None::<SubState>, false);
        world.flush_commands();
        world.run_schedule(StateTransition);

        assert_current_state(&mut world, None::<ManualState>);
        assert_current_state(&mut world, None::<ComputedState>);
        assert_current_state(&mut world, None::<SubState>);

        world.set_state(Some(ManualState::B), None);
        world.flush_commands();
        world.run_schedule(StateTransition);

        assert_current_state(&mut world, Some(ManualState::B));
        assert_current_state(&mut world, Some(ComputedState));
        assert_current_state(&mut world, None::<SubState>);

        world.set_state(Some(ManualState::C), None);
        world.flush_commands();
        world.run_schedule(StateTransition);

        assert_current_state(&mut world, Some(ManualState::C));
        assert_current_state(&mut world, None::<ComputedState>);
        assert_current_state(&mut world, Some(SubState::X));

        world.set_state(Some(SubState::Y), None);
        world.flush_commands();
        world.run_schedule(StateTransition);

        assert_current_state(&mut world, Some(ManualState::C));
        assert_current_state(&mut world, None::<ComputedState>);
        assert_current_state(&mut world, Some(SubState::Y));

        world.set_state(None::<ManualState>, None);
        world.flush_commands();
        world.run_schedule(StateTransition);

        assert_current_state(&mut world, None::<ManualState>);
        assert_current_state(&mut world, None::<ComputedState>);
        assert_current_state(&mut world, None::<SubState>);
    }
}
