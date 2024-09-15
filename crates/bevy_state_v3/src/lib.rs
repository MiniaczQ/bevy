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
        system::{Commands, ResMut, Resource},
        world::World,
    };

    use crate::{
        commands::StatesExt,
        data::{StateData, StateUpdateCurrent},
        events::{StateEnter, StateExit},
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
        world.init_state::<ManualState>(local, true);
        world.init_state::<ComputedState>(local, true);
        world.init_state::<SubState>(local, true);
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

    #[derive(Default, Resource)]
    struct StateTransitionTracker(Vec<&'static str>);

    fn track<E: Event>(s: &'static str) -> impl Fn(Trigger<E>, ResMut<StateTransitionTracker>) {
        move |_: Trigger<E>, mut reg: ResMut<StateTransitionTracker>| {
            reg.0.push(s);
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

        fn update<'a>(
            state: StateUpdateCurrent<Self>,
            dependencies: <<Self as State>::DependencySet as StateSet>::UpdateDependencies<'a>,
        ) -> Option<Option<Self>> {
            let (manual1, manual2) = dependencies;
            match (manual1.current, manual2.current, state.target) {
                (Some(ManualState::B), Some(ManualState2::D), Some(Some(next))) => Some(Some(next)),
                (Some(ManualState::B), Some(ManualState2::D), Some(None)) => None,
                (Some(ManualState::B), Some(ManualState2::D), None) => {
                    Some(Some(SubState2::default()))
                }
                _ => Some(None),
            }
        }
    }

    impl State for ManualState2 {
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

    #[test]
    fn transition_order() {
        let mut world = World::new();
        world.init_resource::<Schedules>();
        world.register_state::<ManualState>();
        world.register_state::<ManualState2>();
        world.register_state::<SubState2>();
        world.register_state::<ComputedState>();
        world.init_state::<ManualState>(None, true);
        world.init_state::<ManualState>(None, true);
        world.init_state::<SubState2>(None, true);
        world.init_state::<ComputedState>(None, true);
        world.next_state(None, Some(ManualState::A));
        world.next_state(None, Some(ManualState2::C));
        world.run_schedule(StateTransition);

        world.init_resource::<StateTransitionTracker>();
        world.observe(track::<StateExit<ManualState>>("m1 ex"));
        world.observe(track::<StateEnter<ManualState>>("m1 en"));
        world.observe(track::<StateExit<ManualState2>>("m2 ex"));
        world.observe(track::<StateEnter<ManualState2>>("m2 en"));
        world.observe(track::<StateExit<SubState2>>("s2 ex"));
        world.observe(track::<StateEnter<SubState2>>("s2 en"));
        world.observe(track::<StateExit<ComputedState>>("c1 ex"));
        world.observe(track::<StateEnter<ComputedState>>("c1 en"));
        world.next_state(None, Some(ManualState::B));
        world.next_state(None, Some(ManualState2::D));
        world.run_schedule(StateTransition);

        let transitions = &world.resource::<StateTransitionTracker>().0;
        assert!(transitions[0..=1].contains(&"c1 ex"));
        assert!(transitions[0..=1].contains(&"s2 ex"));
        assert!(transitions[2..=3].contains(&"m1 ex"));
        assert!(transitions[2..=3].contains(&"m2 ex"));
        assert!(transitions[4..=5].contains(&"m1 en"));
        assert!(transitions[4..=5].contains(&"m2 en"));
        assert!(transitions[6..=7].contains(&"c1 en"));
        assert!(transitions[6..=7].contains(&"s2 en"));
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

    fn log_on_exit<S: State>(
        local: Option<Entity>,
        state: &StateData<S>,
        _commands: &mut Commands,
    ) {
        let pre = if let Some(entity) = local {
            format!("{:?}", entity)
        } else {
            "global".to_owned()
        };
        println!(
            "exit {} {} {:?} -> {:?}",
            pre,
            state_name::<S>(),
            state.previous(),
            state.current()
        );
    }

    fn log_on_enter<S: State>(
        local: Option<Entity>,
        state: &StateData<S>,
        _commands: &mut Commands,
    ) {
        let pre = if let Some(entity) = local {
            format!("{:?}", entity)
        } else {
            "global".to_owned()
        };
        println!(
            "enter {} {} {:?} -> {:?}",
            pre,
            state_name::<S>(),
            state.previous(),
            state.current()
        );
    }
}
