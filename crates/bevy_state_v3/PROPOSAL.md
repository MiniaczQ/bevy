Hey everyone, I've recently been contributing to the states machinery of Bevy.
We still had some refactors & features planned and in the meantime observers and required components landed.
This opened up a lot of possibilities, but also lots of breaking changes, hence I'm opening a discussion for `bevy_states` replacement.
This crate will be developed separately and swapped into Bevy at some point in the future to do all the breaking changes at once.

[Check out the prototype](https://github.com/MiniaczQ/bevy/tree/state-v3/crates/bevy_state_v3/src)

# States v3

States v3 proposal focuses on the following:
- unification of state traits (freely mutable, computed, sub),
- making commands the standard way of updating state,
- migration of state resources to entities,
- enabling per entity state machines while retaining a global state machine,
- cleaning up edge cases.

Something this proposal does not consider is keeping compatibility with the existing crate.

# Internal Design

## Storage Model
A single state machine is stored in a single entity.
Each state entity consists of one or more `StateData<S>` components with different `S: State`.
Dependencies between states are enforced through required components.
The only difference between global and local state is the addition of `GlobalStateMarker` component.
Multiple entites with `GlobalStateMarker` cannot exist.

```rs
/// State data component.
struct StateData<S: State> {
    /// Whether this state was reentered.
    /// Use in tandem with [`previous`].
    is_reentrant: bool,
    /// Last different state value.
    /// This is not overwritten during reentries.
    previous: Option<S>,
    /// Current value of the state.
    current: Option<S>,
    /// Proposed state value to be considered during next [`StateTransition`].
    /// How this value actually impacts the state depends on the [`State::update`] function.
    /// Most often this will be [`StateUpdate`].
    target: S::Target,
    /// Whether this state was updated in the last [`StateTransition`] schedule.
    /// For a standard use case, this happens once per frame.
    updated: bool,
}

// Not derived as to register required components.
impl<S: State> Component for StateData<S> { ... }
```



## Top Level Reactivity
How states update each other.

State machines update through a similar system to existing `bevy_state`.
The `StateTransition` schedule runs 3 system sets:
- `AllUpdates` - Updates based on `target` and dependency changes from root states to leaf states, sets the `updated` flag.
- `AllExits` - Triggers `StateExit<S>` observers from leaf states to root states, targeted for local state, untargeted for global state.
- `AllEnters` - Triggers `StateEnter<S>` observers from root states to leaf states, targeted for local state, untargeted for global state.
Smaller sets are used to specify order in the grap.
Order is derived when specifying state dependencies, smaller value meaning closer to root.

```rs
trait State {
    /// Dependencies of this state.
    type DependencySet: StateSet;

    /// Order of this state.
    const ORDER: u32 = Self::DependencySet::HIGHEST_ORDER + 1;
}

#[derive(SystemSet)]
enum StateSystemSet {
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

/// Emitted in from leaf to root order.
/// Untargeted for global state.
#[derive(Event)]
pub struct StateExit<S: State>(PhantomData<S>);

/// Emitted in from root to leaf order.
/// Untargeted for global state.
#[derive(Event)]
pub struct StateEnter<S: State>(PhantomData<S>);
```

Note that the `Exit` and `Enter` events are called after the entire state machine was updated.
Calling them while state machine is updating can lead to invalid arrangements, which we don't want to expose.
The previous and current values are available in `StateData<S>`.
Based on those events, more user oriented events can be made such as `OnExit { previous, current }` and `OnEnter { previous, current }`.

There is no longer a `Transition` stage inbetween `Exit` and `Enter`.
It wasn't that helpful, the order of calls was undefined and if it was, it could rely on `Exit` or `Enter` instead.



## Bottom level reactivity
How states update themselves.

State change can be triggered from two sources:
- one of the state dependencies changed,
- state `target` has changed.
When any condition is met, the `State::update` function is called.
The return value of this function decides whether and how we update this state.

```rs
trait State {
    type Target: StateTarget = StateUpdate<Self>;

    /// How next value of state is decided.
    fn update(
        // Current state.
        state: &mut StateData<Self>,
        // Dependencies.
        dependencies: (StateUpdateDependency<D1>, StateUpdateDependency<D2>, ...),
    ) -> StateUpdate<Self>;
}

pub trait StateTarget: Default + Send + Sync + 'static {
    /// Returns whether the state should be updated.
    fn should_update(&self) -> bool;

    /// Resets the target to reset change detection.
    fn reset(&mut self);
}

pub enum StateUpdate<S> {
    Nothing,
    Disable,
    Enable(S),
}
```



# User Interface

## Adding and modifying state

First, the `StatesPlugin` has to be added to the application.
This plugin only adds the `StateTransition` schedule to both, startup and main schedules.

```rs
pub trait StatesExt {
    /// Registers machinery for this state as well as all dependencies.
    fn register_state<S: State>(&mut self);
    
    /// Adds the state to the provided `local` entity or otherwise the global state.
    /// If initial update is suppresed, no initial transitions will be generated.
    /// The state added this way is always disabled and has to be enabled through [`next_state`] method.
    /// This also adds all dependencies through required components.
    fn init_state<S: State>(&mut self, local: Option<Entity>, suppress_initial_update: bool);

    /// Sets the [`target`] value in [`StateData`],
    /// which will result in an [`update`] call during [`StateTransition`] schedule.
    /// Much like [`init_state`] you need to provide a local entity or nothing, for global state.
    ///
    /// This only works with the [`StateUpdate`] target.
    fn state_target<S: State<Target = StateUpdate<S>>>(&mut self, local: Option<Entity>, target: Option<S>);
}
```

All of the operations can happen immediatelly (with `World`, `SubApp`, `App`) or in a deferred manner (with `Commands`).
For global state, the entity gets created if it doesn't exist.
For immediate state transitions, running `StateTransition` is recommended.
No helper method is provided due to the complications this can bring.



# Future extensions

There are few harder and easier additions:
- cross-entity states - state machines spread between many entities,
- custom target types - different backends, stacks, directly mutable, etc.,
- state unregistering - entirely removing the state machinery when it's no longer needed.
