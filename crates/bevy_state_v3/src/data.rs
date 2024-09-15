use bevy_ecs::{
    component::{Component, Components, RequiredComponents, StorageType},
    storage::Storages,
};

use crate::state::{State, StateSet};

/// State data component.
#[derive(Debug)]
pub struct StateData<S: State> {
    /// Whether this state was reentered.
    /// Use in tandem with [`Self::previous`].
    pub(crate) is_reentrant: bool,
    /// Last different state value.
    /// This is not overwritten during reentries.
    pub(crate) previous: Option<S>,
    /// Current value of the state.
    pub(crate) current: Option<S>,
    /// Proposed state value to be considered during next [`StateTransition`](crate::state::StateTransition).
    /// How this value actually impacts the state depends on the [`State::update`] function.
    pub(crate) target: S::Target,
    /// Whether this state was updated in the last [`StateTransition`] schedule.
    /// For a standard use case, this happens once per frame.
    pub(crate) is_updated: bool,
}

impl<S: State> Default for StateData<S> {
    fn default() -> Self {
        Self {
            is_reentrant: false,
            previous: None,
            current: None,
            target: S::Target::default(),
            is_updated: false,
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
    pub(crate) fn new(initial: Option<S>, suppress_initial_update: bool) -> Self {
        Self {
            current: initial,
            is_updated: !suppress_initial_update,
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
        self.is_updated = true;
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

    /// Returns the previous state with reentries included.
    pub fn reentrant_previous(&self) -> Option<&S> {
        if self.is_reentrant {
            self.current()
        } else {
            self.previous()
        }
    }

    /// Returns whether the current state was reentered.
    pub fn is_reentrant(&self) -> bool {
        self.is_reentrant
    }

    /// Returns whether the current state was updated last state transition.
    pub fn is_updated(&self) -> bool {
        self.is_updated
    }

    /// Reference to the target.
    pub fn target(&self) -> &S::Target {
        &self.target
    }

    /// Mutable reference to the target.
    pub fn target_mut(&mut self) -> &mut S::Target {
        &mut self.target
    }
}
