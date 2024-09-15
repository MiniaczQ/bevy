//! New states wahoo

#[cfg(feature = "bevy_app")]
mod app;
mod commands;
mod data;
mod events;
mod state;

#[cfg(test)]
mod tests {
    use std::{any::type_name, fmt::Debug};

    use bevy_ecs::{
        entity::Entity,
        event::Event,
        observer::Trigger,
        schedule::Schedules,
        system::{ResMut, Resource},
        world::World,
    };

    use crate::{
        commands::StatesExt,
        data::StateData,
        events::{StateEnter, StateExit},
        state::{State, StateDependencies, StateTransition, StateUpdate},
    };

    #[derive(Clone, Debug, PartialEq)]
    enum ManualState {
        A,
        B,
    }

    impl State for ManualState {
        type DependencySet = ();
        type Target = StateUpdate<Self>;

        fn update<'a>(
            state: &mut StateData<Self>,
            _dependencies: StateDependencies<'_, Self>,
        ) -> StateUpdate<Self> {
            // Pure manual control.
            // We ignore the update call from dependencies, because there are none.
            state.target_mut().take()
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ComputedState;

    impl State for ComputedState {
        type DependencySet = ManualState;
        type Target = StateUpdate<Self>;

        fn update<'a>(
            state: &mut StateData<Self>,
            dependencies: StateDependencies<'_, Self>,
        ) -> StateUpdate<Self> {
            let manual = dependencies;
            match (manual.current(), state.target()) {
                // If next was requested, ignore it.
                (_, StateUpdate::Enable(_)) => StateUpdate::Nothing,
                // If parent is valid, enable the state.
                (Some(ManualState::A), _) => StateUpdate::Enable(ComputedState),
                // If parent is invalid, disable the state.
                _ => StateUpdate::Disable,
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
        type Target = StateUpdate<Self>;

        fn update<'a>(
            state: &mut StateData<Self>,
            dependencies: StateDependencies<'_, Self>,
        ) -> StateUpdate<Self> {
            let manual = dependencies;
            match (manual.current(), state.target_mut().take()) {
                // If parent state is valid, respect requested enabled next state.
                (Some(ManualState::B), StateUpdate::Enable(next)) => StateUpdate::Enable(next),
                // If parent state is valid and requested next state disables the state, ignore it.
                (Some(ManualState::B), StateUpdate::Disable) => StateUpdate::Nothing,
                // If parent state is valid and there was no next request, enable the state with default value.
                (Some(ManualState::B), StateUpdate::Nothing) => {
                    StateUpdate::Enable(SubState::default())
                }
                // If parent state is invalid, disable the state.
                _ => StateUpdate::Disable,
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
        world.init_state::<ManualState>(local, None, true);
        world.init_state::<ComputedState>(local, None, true);
        world.init_state::<SubState>(local, None, true);
        assert_states!(
            world,
            None::<ManualState>,
            None::<ComputedState>,
            None::<SubState>
        );

        world.state_target(local, Some(ManualState::A));
        world.run_schedule(StateTransition);
        assert_states!(
            world,
            Some(ManualState::A),
            Some(ComputedState),
            None::<SubState>
        );

        world.state_target(local, Some(ManualState::B));
        world.run_schedule(StateTransition);
        assert_states!(
            world,
            Some(ManualState::B),
            None::<ComputedState>,
            Some(SubState::X)
        );

        world.state_target(local, Some(SubState::Y));
        world.run_schedule(StateTransition);
        assert_states!(
            world,
            Some(ManualState::B),
            None::<ComputedState>,
            Some(SubState::Y)
        );

        world.state_target(local, None::<ManualState>);
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

    #[derive(Default, Resource)]
    struct StateTransitionTracker(Vec<&'static str>);

    fn track<E: Event>() -> impl Fn(Trigger<E>, ResMut<StateTransitionTracker>) {
        move |_: Trigger<E>, mut reg: ResMut<StateTransitionTracker>| {
            reg.0.push(type_name::<E>());
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    enum ManualState2 {
        C,
        D,
    }

    #[derive(Clone, Debug, Default, PartialEq)]
    enum SubState2 {
        #[default]
        X,
        Y,
    }

    impl State for SubState2 {
        type DependencySet = (ManualState, ManualState2);
        type Target = StateUpdate<Self>;

        fn update<'a>(
            state: &mut StateData<Self>,
            dependencies: StateDependencies<'_, Self>,
        ) -> StateUpdate<Self> {
            let (manual1, manual2) = dependencies;
            match (
                manual1.current(),
                manual2.current(),
                state.target_mut().take(),
            ) {
                (Some(ManualState::B), Some(ManualState2::D), StateUpdate::Enable(next)) => {
                    StateUpdate::Enable(next)
                }
                (Some(ManualState::B), Some(ManualState2::D), StateUpdate::Disable) => {
                    StateUpdate::Nothing
                }
                (Some(ManualState::B), Some(ManualState2::D), StateUpdate::Nothing) => {
                    StateUpdate::Enable(SubState2::X)
                }
                _ => StateUpdate::Disable,
            }
        }
    }

    impl State for ManualState2 {
        type DependencySet = ();
        type Target = StateUpdate<Self>;

        fn update<'a>(
            state: &mut StateData<Self>,
            _dependencies: StateDependencies<'_, Self>,
        ) -> StateUpdate<Self> {
            // Pure manual control.
            // We ignore the update call from dependencies, because there are none.
            state.target_mut().take()
        }
    }

    #[test]
    fn transition_order() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>();
        world.register_state::<ManualState2>();
        world.register_state::<SubState2>();
        world.register_state::<ComputedState>();
        world.init_state::<ManualState>(None, None, true);
        world.init_state::<ManualState>(None, None, true);
        world.init_state::<SubState2>(None, None, true);
        world.init_state::<ComputedState>(None, None, true);
        world.state_target(None, Some(ManualState::A));
        world.state_target(None, Some(ManualState2::C));
        world.run_schedule(StateTransition);

        world.init_resource::<StateTransitionTracker>();
        world.observe(track::<StateExit<ManualState>>());
        world.observe(track::<StateEnter<ManualState>>());
        world.observe(track::<StateExit<ManualState2>>());
        world.observe(track::<StateEnter<ManualState2>>());
        world.observe(track::<StateExit<SubState2>>());
        world.observe(track::<StateEnter<SubState2>>());
        world.observe(track::<StateExit<ComputedState>>());
        world.observe(track::<StateEnter<ComputedState>>());
        world.state_target(None, Some(ManualState::B));
        world.state_target(None, Some(ManualState2::D));
        world.run_schedule(StateTransition);

        let transitions = &world.resource::<StateTransitionTracker>().0;
        assert!(transitions[0..=1].contains(&type_name::<StateExit<SubState2>>()));
        assert!(transitions[0..=1].contains(&type_name::<StateExit<ComputedState>>()));
        assert!(transitions[2..=3].contains(&type_name::<StateExit<ManualState>>()));
        assert!(transitions[2..=3].contains(&type_name::<StateExit<ManualState2>>()));
        assert!(transitions[4..=5].contains(&type_name::<StateEnter<ManualState>>()));
        assert!(transitions[4..=5].contains(&type_name::<StateEnter<ManualState2>>()));
        assert!(transitions[6..=7].contains(&type_name::<StateEnter<SubState2>>()));
        assert!(transitions[6..=7].contains(&type_name::<StateEnter<ComputedState>>()));
    }

    // Debug stuff

    #[allow(unused_macros)]
    macro_rules! print_states {
        ($world:expr, $($state:ty),+) => {
            $(println!("{:?}", $world.query::<&StateData<$state>>().single($world)));+
        };
    }
}
