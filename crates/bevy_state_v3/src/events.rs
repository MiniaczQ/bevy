use std::marker::PhantomData;

use bevy_ecs::event::Event;

use crate::state::State;

/// Triggered when `next` is set during [`StateTransition`](crate::state::StateTransition) or when a dependency transitions.
#[derive(Event)]
pub struct OnUpdate<S: State>(PhantomData<S>);

impl<S: State> Default for OnUpdate<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// Triggered when state transitions.
#[derive(Event)]
pub struct OnTransition<S: State>(PhantomData<S>);

impl<S: State> Default for OnTransition<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}
