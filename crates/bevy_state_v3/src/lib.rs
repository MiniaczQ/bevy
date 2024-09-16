//! New states wahoo

#[cfg(feature = "bevy_app")]
mod app;
mod commands;
mod data;
mod state;
mod state_set;
mod transitions;

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
    use bevy_state_macros_v3::State;

    use crate::{
        self as bevy_state,
        data::StateUpdate,
        state_set::StateDependencies,
        transitions::{OnEnter, OnExit, StateTransitionsConfig},
    };
    use crate::{
        commands::StatesExt,
        data::StateData,
        state::{State, StateTransition},
    };

    #[derive(State, Clone, Debug, PartialEq)]
    enum ManualState {
        A,
        B,
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ComputedState;

    impl State for ComputedState {
        type DependencySet = ManualState;
        type Target = ();

        fn update<'a>(
            _state: &mut StateData<Self>,
            dependencies: StateDependencies<'_, Self>,
        ) -> StateUpdate<Self> {
            let manual = dependencies;
            match manual.current() {
                // If parent is valid, enable the state.
                Some(ManualState::A) => StateUpdate::Enable(ComputedState),
                // If parent is invalid, disable the state.
                _ => StateUpdate::Disable,
            }
        }
    }

    #[derive(State, Clone, Debug, Default, PartialEq)]
    #[dependency(ManualState = ManualState::B)]
    enum SubState {
        #[default]
        X,
        Y,
    }

    macro_rules! assert_states {
        ($world:expr, $($state:expr),+) => {
            $(assert_eq!($world.query::<&StateData<_>>().single($world).current, $state));+
        };
    }

    fn test_all_states(world: &mut World, local: Option<Entity>) {
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>(StateTransitionsConfig::empty());
        world.register_state::<ComputedState>(StateTransitionsConfig::empty());
        world.register_state::<SubState>(StateTransitionsConfig::empty());
        world.register_state::<SubState>(StateTransitionsConfig::empty());
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

    #[derive(State, Clone, Debug, PartialEq)]
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

    #[test]
    fn transition_order() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>(StateTransitionsConfig::default());
        world.register_state::<ManualState2>(StateTransitionsConfig::default());
        world.register_state::<SubState2>(StateTransitionsConfig::default());
        world.register_state::<ComputedState>(StateTransitionsConfig::default());
        world.init_state::<ManualState>(None, None, true);
        world.init_state::<ManualState>(None, None, true);
        world.init_state::<SubState2>(None, None, true);
        world.init_state::<ComputedState>(None, None, true);
        world.state_target(None, Some(ManualState::A));
        world.state_target(None, Some(ManualState2::C));
        world.run_schedule(StateTransition);

        world.init_resource::<StateTransitionTracker>();
        world.observe(track::<OnExit<ManualState>>());
        world.observe(track::<OnEnter<ManualState>>());
        world.observe(track::<OnExit<ManualState2>>());
        world.observe(track::<OnEnter<ManualState2>>());
        world.observe(track::<OnExit<SubState2>>());
        world.observe(track::<OnEnter<SubState2>>());
        world.observe(track::<OnExit<ComputedState>>());
        world.observe(track::<OnEnter<ComputedState>>());
        world.state_target(None, Some(ManualState::B));
        world.state_target(None, Some(ManualState2::D));
        world.run_schedule(StateTransition);

        let transitions = &world.resource::<StateTransitionTracker>().0;
        // Test in groups, because order of directly unrelated states is non-deterministic.
        assert!(transitions[0..=1].contains(&type_name::<OnExit<SubState2>>()));
        assert!(transitions[0..=1].contains(&type_name::<OnExit<ComputedState>>()));
        assert!(transitions[2..=3].contains(&type_name::<OnExit<ManualState>>()));
        assert!(transitions[2..=3].contains(&type_name::<OnExit<ManualState2>>()));
        assert!(transitions[4..=5].contains(&type_name::<OnEnter<ManualState>>()));
        assert!(transitions[4..=5].contains(&type_name::<OnEnter<ManualState2>>()));
        assert!(transitions[6..=7].contains(&type_name::<OnEnter<SubState2>>()));
        assert!(transitions[6..=7].contains(&type_name::<OnEnter<ComputedState>>()));
    }

    // Debug stuff

    #[allow(unused_macros)]
    macro_rules! print_states {
        ($world:expr, $($state:ty),+) => {
            $(println!("{:?}", $world.query::<&StateData<$state>>().single($world)));+
        };
    }
}
