use bevy_ecs::{
    component::{Component, Components, RequiredComponents, StorageType},
    storage::Storages,
};

use crate::state::{State, StateSet};

/// State data.
#[derive(Debug)]
pub struct StateData<S: State> {
    /// Counter of reentrant transitions for the [`Self::current`] state.
    /// If this value were to overflow, it wraps.
    pub(crate) reenters: usize,
    /// This is the previous (different) state.
    /// In case of reentries check [`Self::reenters`].
    pub(crate) previous: Option<S>,
    /// Current state.
    pub(crate) current: Option<S>,
    /// Next requested state.
    /// Note that this value is processed through [`State::on_update`] as opposed to directly mutating [`Self::current`].
    pub(crate) next: Option<Option<S>>,
}

impl<S: State> Default for StateData<S> {
    fn default() -> Self {
        Self {
            reenters: 0,
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
            self.reenters = self.reenters.wrapping_add(1);
        } else {
            self.reenters = 0;
            self.previous = self.current.take();
            self.current = next;
        }
    }
}

impl<S: State> StateData<S> {
    /// Returns the [`StateData<S>::current`].
    pub fn get_current(&self) -> Option<&S> {
        self.current.as_ref()
    }

    /// Returns the [`StateData<S>::previous`].
    pub fn get_previous(&self) -> Option<&S> {
        self.previous.as_ref()
    }

    /// Returns the [`StateData<S>::next`].
    pub fn get_next(&self) -> Option<Option<&S>> {
        self.next.as_ref().map(Option::as_ref)
    }

    /// Returns the [`StateData<S>::reenters`].
    pub fn reenters(&self) -> usize {
        self.reenters
    }
}
