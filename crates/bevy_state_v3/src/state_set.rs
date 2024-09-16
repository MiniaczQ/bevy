use std::u32;

use bevy_ecs::{
    component::{Components, RequiredComponents},
    query::{ReadOnlyQueryData, WorldQuery},
    storage::Storages,
    world::World,
};
use bevy_utils::all_tuples;

use crate::{data::StateData, state::State, transitions::StateTransitionsConfig};

pub type StateDependencies<'a, S> =
    <<<S as State>::DependencySet as StateSet>::Query as WorldQuery>::Item<'a>;

/// Set of states used for dependencies.
pub trait StateSet {
    /// Parameters provided to [`State::on_update`].
    type Query: ReadOnlyQueryData;

    const HIGHEST_ORDER: u32;

    /// Registers all elements as required components.
    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    );

    /// Registers all required states.
    fn register_required_states(world: &mut World);

    /// Check dependencies for changes.
    fn is_changed(set: &<Self::Query as WorldQuery>::Item<'_>) -> bool;
}

impl StateSet for () {
    type Query = ();

    const HIGHEST_ORDER: u32 = 0;

    fn register_required_components(
        _components: &mut Components,
        _storages: &mut Storages,
        _required_components: &mut RequiredComponents,
    ) {
    }

    fn register_required_states(_world: &mut World) {}

    fn is_changed(_set: &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        false
    }
}

impl<S1: State> StateSet for S1 {
    type Query = &'static StateData<S1>;

    const HIGHEST_ORDER: u32 = S1::ORDER;

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        required_components.register(components, storages, StateData::<S1>::default);
    }

    fn register_required_states(world: &mut World) {
        S1::register_state(world, StateTransitionsConfig::default(), true);
    }

    fn is_changed(s1: &<Self::Query as WorldQuery>::Item<'_>) -> bool {
        s1.is_updated
    }
}

const fn const_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

macro_rules! max {
    ($a:expr) => ( $a );
    ($a:expr, $b:expr) => {
        const_max($a, $b)
    };
    ($a:expr, $b:expr, $($other:expr), *) => {
        max!(const_max($a, $b), $($other), +)
    };
}

macro_rules! impl_state_set {
    ($(#[$meta:meta])* $(($type:ident, $var:ident)), *) => {
        $(#[$meta])*
        impl<$($type: State), *> StateSet for ($($type, )*) {
            type Query = ($(&'static StateData<$type>, )*);

            const HIGHEST_ORDER: u32 = max!($($type::ORDER), +);

            fn register_required_components(
                components: &mut Components,
                storages: &mut Storages,
                required_components: &mut RequiredComponents,
            ) {
                $(required_components.register(components, storages, StateData::<$type>::default);)
                +
            }

            fn register_required_states(world: &mut World) {
                $($type::register_state(world, StateTransitionsConfig::default(), true);)
                +
            }

            fn is_changed(($($var, )+): &<Self::Query as WorldQuery>::Item<'_>) -> bool {
                $($var.is_updated) || +
            }
        }
    };
}

all_tuples!(
    #[doc(fake_variadic)]
    impl_state_set,
    1,
    15,
    S,
    s
);
