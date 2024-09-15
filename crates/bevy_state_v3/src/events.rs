use std::marker::PhantomData;

use bevy_ecs::event::Event;

use crate::state::State;

#[derive(Event)]
pub struct StateExit<S: State>(PhantomData<S>);

impl<S: State> Default for StateExit<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Event)]
pub struct StateEnter<S: State>(PhantomData<S>);

impl<S: State> Default for StateEnter<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}
