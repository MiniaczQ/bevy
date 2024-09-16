use bevy_ecs::schedule::{IntoSystemSetConfigs, ScheduleLabel, SystemSet};

use crate::state::State;

/// State transition schedule.
/// All states are updated and their events are emitted during this schedule.
/// You can run this schedule by hand to perform state transition at different intervals.
/// The [`StatesPlugin`](crate::app::StatesPlugin) adds this transition before [`PreStartup`](bevy_app::PreStartup) and [`PreUpdate`](bevy_app::PreUpdate).
#[derive(ScheduleLabel, Debug, PartialEq, Eq, Hash, Clone)]
pub struct StateTransition;

#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
/// The `StateTransition` schedule runs 3 system sets:
/// - [`AllUpdates`] - Updates based on `target` and dependency changes from root states to leaf states, sets the `updated` flag.
/// - [`AllExits`] - Triggers [`StateExit<S>`] observers from leaf states to root states, targeted for local state, untargeted for global state.
/// - [`AllEnters`] - Triggers [`StateEnter<S>`] observers from root states to leaf states, targeted for local state, untargeted for global state.
/// Smaller sets are used to specify order in the grap.
/// Order is derived when specifying state dependencies, smaller value meaning closer to root.
pub enum StateSystemSet {
    /// All [`Update`]s.
    AllUpdates,
    /// Lower values before higher ones.
    Update(u32),
    /// All [`Exit`]s.
    AllExits,
    /// Higher values then lower ones.
    Exit(u32),
    /// All [`Enter`]s.
    AllEnters,
    /// Same as [`Update`], lower values before higher ones.
    Enter(u32),
}

impl StateSystemSet {
    pub fn update<S: State>() -> Self {
        Self::Update(S::ORDER)
    }

    pub fn exit<S: State>() -> Self {
        Self::Exit(S::ORDER)
    }

    pub fn enter<S: State>() -> Self {
        Self::Enter(S::ORDER)
    }

    pub fn configuration<S: State>() -> impl IntoSystemSetConfigs {
        (
            (Self::AllUpdates, Self::AllExits, Self::AllEnters).chain(),
            Self::update::<S>()
                .after(Self::Update(S::ORDER - 1))
                .in_set(Self::AllUpdates),
            Self::exit::<S>()
                .before(Self::Exit(S::ORDER - 1))
                .in_set(Self::AllExits),
            Self::enter::<S>()
                .after(Self::Enter(S::ORDER - 1))
                .in_set(Self::AllEnters),
        )
    }
}
