use std::marker::PhantomData;

use bevy_ecs::event::Event;

use crate::state::State;

#[derive(Event)]
pub struct StateUpdate<S: State>(PhantomData<S>);

impl<S: State> Default for StateUpdate<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}
