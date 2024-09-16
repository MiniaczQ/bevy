use bevy_ecs::{
    query::With,
    system::{Query, SystemParam},
    world::Mut,
};

use crate::{
    data::{GlobalStateMarker, StateData},
    state::State,
};

#[derive(SystemParam)]
pub struct GlobalState<'w, 's, S: State> {
    query: Query<'w, 's, &'static StateData<S>, With<GlobalStateMarker>>,
}

impl<'w, 's, S: State> GlobalState<'w, 's, S> {
    pub fn get(&self) -> &StateData<S> {
        self.query.single()
    }

    pub fn try_get(&self) -> Option<&StateData<S>> {
        self.query.get_single().ok()
    }
}

#[derive(SystemParam)]
pub struct GlobalStateMut<'w, 's, S: State> {
    query: Query<'w, 's, &'static mut StateData<S>, With<GlobalStateMarker>>,
}

impl<'w, 's, S: State> GlobalStateMut<'w, 's, S> {
    pub fn get(&self) -> &StateData<S> {
        self.query.single()
    }

    pub fn try_get(&self) -> Option<&StateData<S>> {
        self.query.get_single().ok()
    }

    pub fn get_mut(&mut self) -> Mut<StateData<S>> {
        self.query.single_mut()
    }

    pub fn try_get_mut(&mut self) -> Option<Mut<StateData<S>>> {
        self.query.get_single_mut().ok()
    }
}

/// Run condition.
/// Returns true if global state is set to this value.
pub fn in_state<S: State>(target: Option<S>) -> impl Fn(GlobalState<S>) -> bool {
    move |state: GlobalState<S>| state.get().current() == target.as_ref()
}
