use std::marker::PhantomData;

use bevy_ecs::event::Event;

use crate::state::State;

/// Triggered when `next` is set during [`StateTransition`](crate::state::StateTransition) or when a dependency transitions.
#[derive(Event)]
pub struct OnStateUpdate<S: State>(PhantomData<S>);

impl<S: State> Default for OnStateUpdate<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// Triggered when state transitions.
#[derive(Event)]
pub struct OnStateTransition<S: State>(PhantomData<S>);

impl<S: State> Default for OnStateTransition<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}
