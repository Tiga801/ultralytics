"""Task state management.

This module provides the TaskState enum and TaskStateMachine for managing
task lifecycle states with valid transitions.
"""

import threading
from enum import Enum, auto
from typing import Optional, Set


class TaskState(Enum):
    """Task lifecycle states.

    State diagram:
        INIT -> RUNNING -> PAUSED -> RUNNING -> STOPPED
                       -> STOPPED
                       -> ERROR -> STOPPED
    """

    INIT = auto()       # Task created but not started
    RUNNING = auto()    # Task is actively processing
    PAUSED = auto()     # Task is temporarily suspended
    STOPPED = auto()    # Task is terminated (terminal state)
    ERROR = auto()      # Task encountered an error

    def __str__(self) -> str:
        """Return lowercase state name."""
        return self.name.lower()


class TaskStateMachine:
    """Thread-safe task state machine with valid transition enforcement.

    This class manages task state transitions and ensures that only valid
    state changes are allowed.

    Valid transitions:
        INIT -> RUNNING
        RUNNING -> PAUSED, STOPPED, ERROR
        PAUSED -> RUNNING, STOPPED
        ERROR -> STOPPED
        STOPPED -> (none, terminal state)

    Example:
        >>> sm = TaskStateMachine()
        >>> sm.state
        <TaskState.INIT: 1>
        >>> sm.transition(TaskState.RUNNING)
        True
        >>> sm.transition(TaskState.PAUSED)
        True
        >>> sm.transition(TaskState.RUNNING)
        True
        >>> sm.transition(TaskState.STOPPED)
        True
    """

    VALID_TRANSITIONS: dict[TaskState, Set[TaskState]] = {
        TaskState.INIT: {TaskState.RUNNING},
        TaskState.RUNNING: {TaskState.PAUSED, TaskState.STOPPED, TaskState.ERROR},
        TaskState.PAUSED: {TaskState.RUNNING, TaskState.STOPPED},
        TaskState.ERROR: {TaskState.STOPPED},
        TaskState.STOPPED: set(),  # Terminal state, no transitions allowed
    }

    def __init__(self, initial_state: TaskState = TaskState.INIT):
        """Initialize state machine.

        Args:
            initial_state: Initial state (default INIT).
        """
        self._state = initial_state
        self._lock = threading.Lock()
        self._error_message: Optional[str] = None

    @property
    def state(self) -> TaskState:
        """Get current state.

        Returns:
            Current TaskState.
        """
        with self._lock:
            return self._state

    @property
    def error_message(self) -> Optional[str]:
        """Get error message if in ERROR state.

        Returns:
            Error message or None.
        """
        with self._lock:
            return self._error_message

    def can_transition(self, new_state: TaskState) -> bool:
        """Check if transition to new state is valid.

        Args:
            new_state: Target state.

        Returns:
            True if transition is valid.
        """
        with self._lock:
            return new_state in self.VALID_TRANSITIONS.get(self._state, set())

    def transition(
        self,
        new_state: TaskState,
        error_message: Optional[str] = None
    ) -> bool:
        """Attempt to transition to a new state.

        Args:
            new_state: Target state.
            error_message: Optional error message (used when transitioning to ERROR).

        Returns:
            True if transition was successful, False otherwise.
        """
        with self._lock:
            valid_transitions = self.VALID_TRANSITIONS.get(self._state, set())
            if new_state not in valid_transitions:
                return False

            self._state = new_state
            if new_state == TaskState.ERROR:
                self._error_message = error_message
            elif new_state == TaskState.RUNNING:
                self._error_message = None  # Clear error on resume

            return True

    def force_state(self, new_state: TaskState) -> None:
        """Force set state without validation.

        WARNING: Use only for testing or recovery scenarios.

        Args:
            new_state: State to set.
        """
        with self._lock:
            self._state = new_state

    def is_running(self) -> bool:
        """Check if task is in RUNNING state.

        Returns:
            True if running.
        """
        with self._lock:
            return self._state == TaskState.RUNNING

    def is_paused(self) -> bool:
        """Check if task is in PAUSED state.

        Returns:
            True if paused.
        """
        with self._lock:
            return self._state == TaskState.PAUSED

    def is_stopped(self) -> bool:
        """Check if task is in STOPPED state.

        Returns:
            True if stopped.
        """
        with self._lock:
            return self._state == TaskState.STOPPED

    def is_error(self) -> bool:
        """Check if task is in ERROR state.

        Returns:
            True if in error state.
        """
        with self._lock:
            return self._state == TaskState.ERROR

    def is_terminal(self) -> bool:
        """Check if task is in a terminal state.

        Returns:
            True if in STOPPED state.
        """
        with self._lock:
            return self._state == TaskState.STOPPED

    def is_active(self) -> bool:
        """Check if task is in an active (non-terminal) state.

        Returns:
            True if in INIT, RUNNING, PAUSED, or ERROR state.
        """
        with self._lock:
            return self._state != TaskState.STOPPED

    def __repr__(self) -> str:
        """String representation."""
        return f"TaskStateMachine(state={self._state})"
