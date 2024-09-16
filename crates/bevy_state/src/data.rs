use std::marker::PhantomData;

use bevy_ecs::{
    component::{Component, Components, RequiredComponents, StorageType},
    storage::Storages,
};

use crate::{state::State, state_set::StateSet};

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
    /// Creates a new instance with initial value.
    pub fn new(initial: Option<S>, suppress_initial_update: bool) -> Self {
        Self {
            current: initial,
            is_updated: !suppress_initial_update,
            ..Default::default()
        }
    }

    /// Returns the current state.
    pub fn current(&self) -> Option<&S> {
        self.current.as_ref()
    }

    /// Returns the previous, different state.
    /// If the current state was reentered, this value will remain unchanged,
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

/// Marker component for global states.
#[derive(Component)]
pub struct GlobalStateMarker;

/// Used to keep track of which states are registered and which aren't.
#[derive(Component)]
pub struct RegisteredState<S: State>(PhantomData<S>);

impl<S: State> Default for RegisteredState<S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Default, Debug, Clone)]
pub enum StateUpdate<S> {
    #[default]
    Nothing,
    Disable,
    Enable(S),
}

impl<S> StateUpdate<S> {
    pub fn is_something(&self) -> bool {
        if let Self::Nothing = self {
            false
        } else {
            true
        }
    }

    pub fn as_options(self) -> Option<Option<S>> {
        match self {
            StateUpdate::Nothing => None,
            StateUpdate::Disable => Some(None),
            StateUpdate::Enable(s) => Some(Some(s)),
        }
    }

    pub fn as_ref(&self) -> StateUpdate<&S> {
        match &self {
            StateUpdate::Nothing => StateUpdate::Nothing,
            StateUpdate::Disable => StateUpdate::Disable,
            StateUpdate::Enable(s) => StateUpdate::Enable(s),
        }
    }

    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

/// Variable target backend for states.
/// Different backends can allow for different features:
/// - [`()`] for no manual updates, only dependency based ones (computed states),
/// - [`StateUpdate`] for overwrite-style control (root/sub states),
/// - mutable target state, for combining multiple requests,
/// - stack or vector of states.
pub trait StateTarget: Default + Send + Sync + 'static {
    /// Returns whether the state should be updated.
    fn should_update(&self) -> bool;

    /// Resets the target to reset change detection.
    fn reset(&mut self);
}

impl<S: State> StateTarget for StateUpdate<S> {
    fn should_update(&self) -> bool {
        self.is_something()
    }

    fn reset(&mut self) {
        self.take();
    }
}

impl StateTarget for () {
    fn should_update(&self) -> bool {
        false
    }

    fn reset(&mut self) {}
}
