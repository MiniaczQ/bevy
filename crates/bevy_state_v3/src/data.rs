use bevy_ecs::{
    component::{Component, Components, RequiredComponents, StorageType},
    storage::Storages,
};

use crate::state::{State, StateSet};

/// Data of the state.
#[derive(Debug)]
pub struct StateData<S: State> {
    pub(crate) is_reentrant: bool,
    pub(crate) previous: Option<S>,
    pub(crate) current: Option<S>,
    pub(crate) next: Option<Option<S>>,
}

impl<S: State> Default for StateData<S> {
    fn default() -> Self {
        Self {
            is_reentrant: false,
            previous: None,
            current: None,
            next: None,
        }
    }
}

impl<S: State + Send + Sync + 'static> Component for StateData<S> {
    const STORAGE_TYPE: StorageType = StorageType::Table;

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        <S::Dependencies as StateSet>::register_required_components(
            components,
            storages,
            required_components,
        );
    }
}

impl<S: State> StateData<S> {
    pub(crate) fn new(next: Option<S>) -> Self {
        Self {
            next: Some(next),
            ..Default::default()
        }
    }

    pub(crate) fn advance(&mut self, next: Option<S>) {
        if next == self.current {
            self.is_reentrant = true;
        } else {
            self.is_reentrant = false;
            self.previous = self.current.take();
            self.current = next;
        }
    }
}

impl<S: State> StateData<S> {
    /// Returns the current state.
    pub fn current(&self) -> Option<&S> {
        self.current.as_ref()
    }

    /// Returns the previous, different state.
    /// If the current state was reentered, this value won't be overwritten,
    /// instead the [`Self::is_reentrant()`] flag will be raised.
    pub fn previous(&self) -> Option<&S> {
        self.previous.as_ref()
    }

    /// Next requested state.
    /// Note that this value is processed during [`State::update`] as opposed to directly mutating the current value.
    pub fn next(&self) -> Option<Option<&S>> {
        self.next.as_ref().map(Option::as_ref)
    }

    /// Returns whether the current state was reentered.
    pub fn is_reentrant(&self) -> bool {
        self.is_reentrant
    }
}
