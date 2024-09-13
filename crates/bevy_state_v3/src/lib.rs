//! New states wahoo

mod commands;
mod data;
mod events;
mod state;

#[cfg(test)]
mod tests {
    use bevy_ecs::{query::WorldQuery, schedule::Schedules, world::World};

    use crate::{
        commands::StatesExt,
        data::StateData,
        state::{State, StateSet, StateTransition},
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

    fn assert_only_state<S: State>(world: &mut World, _target: Option<S>) {
        let state = world.query::<&StateData<S>>().single(world);
        assert!(matches!(state.current(), _target));
        assert!(matches!(state.next(), None));
    }

    #[test]
    fn global_state() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>();
        world.register_state::<ComputedState>();
        world.register_state::<SubState>();

        world.insert_state(None, None::<ManualState>, false);
        world.insert_state(None, None::<ComputedState>, false);
        world.insert_state(None, None::<SubState>, false);
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, None::<ManualState>);
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, None::<SubState>);

        world.next_state(None, Some(ManualState::B));
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, Some(ManualState::B));
        assert_only_state(&mut world, Some(ComputedState));
        assert_only_state(&mut world, None::<SubState>);

        world.next_state(None, Some(ManualState::C));
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, Some(ManualState::C));
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, Some(SubState::X));

        world.next_state(None, Some(SubState::Y));
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, Some(ManualState::C));
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, Some(SubState::Y));

        world.next_state(None, None::<ManualState>);
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, None::<ManualState>);
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, None::<SubState>);
    }

    #[test]
    fn local_state() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>();
        world.register_state::<ComputedState>();
        world.register_state::<SubState>();

        let entity = world.spawn_empty().id();
        world.insert_state(Some(entity), None::<ManualState>, false);
        world.insert_state(Some(entity), None::<ComputedState>, false);
        world.insert_state(Some(entity), None::<SubState>, false);
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, None::<ManualState>);
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, None::<SubState>);

        world.next_state(Some(entity), Some(ManualState::B));
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, Some(ManualState::B));
        assert_only_state(&mut world, Some(ComputedState));
        assert_only_state(&mut world, None::<SubState>);

        world.next_state(Some(entity), Some(ManualState::C));
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, Some(ManualState::C));
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, Some(SubState::X));

        world.next_state(Some(entity), Some(SubState::Y));
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, Some(ManualState::C));
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, Some(SubState::Y));

        world.next_state(Some(entity), None::<ManualState>);
        world.run_schedule(StateTransition);

        assert_only_state(&mut world, None::<ManualState>);
        assert_only_state(&mut world, None::<ComputedState>);
        assert_only_state(&mut world, None::<SubState>);
    }
}
