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
    pub(crate) target: Option<Option<S>>,
    pub(crate) updated: bool,
}

impl<S: State> Default for StateData<S> {
    fn default() -> Self {
        Self {
            is_reentrant: false,
            previous: None,
            current: None,
            target: None,
            // Start with `updated` to trigger validation.
            updated: true,
        }
    }
}

impl<S: State> Component for StateData<S> {
    const STORAGE_TYPE: StorageType = StorageType::Table;

    fn register_required_components(
        components: &mut Components,
        storages: &mut Storages,
        required_components: &mut RequiredComponents,
    ) {
        <S::DependencySet as StateSet>::register_required_components(
            components,
            storages,
            required_components,
        );
    }
}

impl<S: State> StateData<S> {
    pub(crate) fn new(suppress_initial_update: bool) -> Self {
        Self {
            updated: !suppress_initial_update,
            ..Default::default()
        }
    }

    pub(crate) fn update(&mut self, next: Option<S>) {
        if next == self.current {
            self.is_reentrant = true;
        } else {
            self.is_reentrant = false;
            self.previous = self.current.take();
            self.current = next;
        }
        self.updated = true;
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

    /// Returns whether the current state was reentered.
    pub fn is_reentrant(&self) -> bool {
        self.is_reentrant
    }

    pub fn target_mut(&mut self) -> &mut Option<Option<S>> {
        &mut self.target
    }
}

pub struct StateUpdateCurrent<'a, S: State> {
    pub current: Option<&'a S>,
    pub target: Option<Option<S>>,
}

impl<'a, S: State> From<&'a StateData<S>> for StateUpdateCurrent<'a, S> {
    fn from(value: &'a StateData<S>) -> Self {
        StateUpdateCurrent {
            current: value.current.as_ref(),
            target: value.target.clone(),
        }
    }
}

pub struct StateUpdateDependency<'a, S: State> {
    pub current: Option<&'a S>,
    pub updated: bool,
}

impl<'a, S: State> From<&'a StateData<S>> for StateUpdateDependency<'a, S> {
    fn from(value: &'a StateData<S>) -> Self {
        StateUpdateDependency {
            current: value.current.as_ref(),
            updated: value.updated,
        }
    }
}
