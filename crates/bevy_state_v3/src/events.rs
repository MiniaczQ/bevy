use bevy_ecs::event::Event;

use crate::state::State;

/// Event triggered when a state is exited.
/// Reentrant transitions are ignored.
#[derive(Event)]
pub struct OnExit<S: State> {
    /// Previous state.
    previous: Option<S>,
    /// Current state.
    current: Option<S>,
}

impl<S: State> OnExit<S> {
    pub fn new(previous: Option<S>, current: Option<S>) -> Self {
        Self { previous, current }
    }
}

/// Event triggered when a state is entered.
/// Reentrant transitions are ignored.
#[derive(Event)]
pub struct OnEnter<S: State> {
    /// Previous state.
    previous: Option<S>,
    /// Current state.
    current: Option<S>,
}

impl<S: State> OnEnter<S> {
    pub fn new(previous: Option<S>, current: Option<S>) -> Self {
        Self { previous, current }
    }
}
