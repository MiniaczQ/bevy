//! New states wahoo

mod commands;
mod data;
mod events;
mod state;

#[cfg(test)]
mod tests {
    use std::{any::type_name, fmt::Debug};

    use bevy_ecs::{entity::Entity, observer::Trigger, schedule::Schedules, world::World};

    use crate::{
        commands::StatesExt,
        data::{StateData, StateUpdateCurrent},
        events::OnStateTransition,
        state::{State, StateSet, StateTransition},
    };

    #[derive(Clone, Debug, PartialEq)]
    enum ManualState {
        A,
        B,
    }

    impl State for ManualState {
        type DependencySet = ();

        fn update<'a>(
            state: StateUpdateCurrent<Self>,
            _dependencies: <<Self as State>::DependencySet as StateSet>::UpdateDependencies<'a>,
        ) -> Option<Option<Self>> {
            // Pure manual control.
            // We ignore the update call from dependencies, because there are none.
            state.target
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ComputedState;

    impl State for ComputedState {
        type DependencySet = ManualState;

        fn update<'a>(
            state: StateUpdateCurrent<Self>,
            dependencies: <<Self as State>::DependencySet as StateSet>::UpdateDependencies<'a>,
        ) -> Option<Option<Self>> {
            let manual = dependencies;
            match (manual.current, state.target) {
                // If next was requested, ignore it.
                (_, Some(_)) => None,
                // If parent is valid, enable the state.
                (Some(ManualState::A), _) => Some(Some(ComputedState)),
                // If parent is invalid, disable the state.
                _ => Some(None),
            }
        }
    }

    #[derive(Clone, Debug, Default, PartialEq)]
    enum SubState {
        #[default]
        X,
        Y,
    }

    impl State for SubState {
        type DependencySet = ManualState;

        fn update<'a>(
            state: StateUpdateCurrent<Self>,
            dependencies: <<Self as State>::DependencySet as StateSet>::UpdateDependencies<'a>,
        ) -> Option<Option<Self>> {
            let manual = dependencies;
            match (manual.current, state.target) {
                // If parent state is valid, respect requested enabled next state.
                (Some(ManualState::B), Some(Some(next))) => Some(Some(next)),
                // If parent state is valid and requested next state disables the state, ignore it.
                (Some(ManualState::B), Some(None)) => None,
                // If parent state is valid and there was no next request, enable the state with default value.
                (Some(ManualState::B), None) => Some(Some(SubState::default())),
                // If parent state is invalid, disable the state.
                _ => Some(None),
            }
        }
    }

    macro_rules! assert_states {
        ($world:expr, $($state:expr),+) => {
            $(assert_eq!($world.query::<&StateData<_>>().single($world).current, $state));+
        };
    }

    fn test_all_states(world: &mut World, local: Option<Entity>) {
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>();
        world.register_state::<ComputedState>();
        world.register_state::<SubState>();
        world.init_state::<ManualState>(local);
        world.init_state::<ComputedState>(local);
        world.init_state::<SubState>(local);

        assert_states!(
            world,
            None::<ManualState>,
            None::<ComputedState>,
            None::<SubState>
        );

        world.next_state(local, Some(ManualState::A));
        world.run_schedule(StateTransition);

        assert_states!(
            world,
            Some(ManualState::A),
            Some(ComputedState),
            None::<SubState>
        );

        world.next_state(local, Some(ManualState::B));
        world.run_schedule(StateTransition);

        assert_states!(
            world,
            Some(ManualState::B),
            None::<ComputedState>,
            Some(SubState::X)
        );

        world.next_state(local, Some(SubState::Y));
        world.run_schedule(StateTransition);

        assert_states!(
            world,
            Some(ManualState::B),
            None::<ComputedState>,
            Some(SubState::Y)
        );

        world.next_state(local, None::<ManualState>);
        world.run_schedule(StateTransition);

        assert_states!(
            world,
            None::<ManualState>,
            None::<ComputedState>,
            None::<SubState>
        );
    }

    #[test]
    fn global_state() {
        let mut world = World::new();
        let local = None;
        test_all_states(&mut world, local);
    }

    #[test]
    fn local_state() {
        let mut world = World::new();
        let local = Some(world.spawn_empty().id());
        test_all_states(&mut world, local);
    }

    #[test]
    fn transition_order() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>();
        world.register_state::<ComputedState>();
        world.register_state::<SubState>();
        world.init_state::<ManualState>(None);
        world.init_state::<ComputedState>(None);
        world.init_state::<SubState>(None);

        world.next_state(None, Some(ManualState::A));
        world.run_schedule(StateTransition);

        world.next_state(None, Some(ManualState::B));
        world.run_schedule(StateTransition);

        world.next_state(None, Some(ManualState::A));
        world.run_schedule(StateTransition);

        // TODO: still working on transitions
    }

    // Debug stuff

    #[allow(unused_macros)]
    macro_rules! print_states {
        ($world:expr, $($state:ty),+) => {
            $(println!("{:?}", $world.query::<&StateData<$state>>().single($world)));+
        };
    }

    #[allow(dead_code)]
    fn state_name<S: State>() -> &'static str {
        type_name::<S>().split("::").last().unwrap()
    }

    #[allow(dead_code)]
    fn log_state<S: State>(world: &mut World) {
        let name = state_name::<S>();
        world.observe(move |_: Trigger<OnStateTransition<S>>| println!("{} - Transition", name));
    }
}
