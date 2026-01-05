"""Tests for TaskState and TaskStateMachine.

This module tests the task state machine including state transitions,
validation, and edge cases.
"""

import pytest
import sys

from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task.state import TaskState, TaskStateMachine


class TestTaskState:
    """Test cases for TaskState enum."""

    def test_state_values(self):
        """Test TaskState enum values."""
        assert TaskState.INIT.value == "init"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.PAUSED.value == "paused"
        assert TaskState.STOPPED.value == "stopped"
        assert TaskState.ERROR.value == "error"

    def test_state_string_representation(self):
        """Test TaskState string representation."""
        assert str(TaskState.INIT) == "init"
        assert str(TaskState.RUNNING) == "running"


class TestTaskStateMachine:
    """Test cases for TaskStateMachine."""

    def test_initial_state(self):
        """Test state machine starts in INIT state."""
        sm = TaskStateMachine()
        assert sm.state == TaskState.INIT

    def test_valid_transition_init_to_running(self):
        """Test valid transition from INIT to RUNNING."""
        sm = TaskStateMachine()
        assert sm.can_transition(TaskState.RUNNING)
        result = sm.transition(TaskState.RUNNING)
        assert result is True
        assert sm.state == TaskState.RUNNING

    def test_valid_transition_running_to_paused(self):
        """Test valid transition from RUNNING to PAUSED."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        assert sm.can_transition(TaskState.PAUSED)
        result = sm.transition(TaskState.PAUSED)
        assert result is True
        assert sm.state == TaskState.PAUSED

    def test_valid_transition_paused_to_running(self):
        """Test valid transition from PAUSED to RUNNING (resume)."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        sm.transition(TaskState.PAUSED)
        assert sm.can_transition(TaskState.RUNNING)
        result = sm.transition(TaskState.RUNNING)
        assert result is True
        assert sm.state == TaskState.RUNNING

    def test_valid_transition_running_to_stopped(self):
        """Test valid transition from RUNNING to STOPPED."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        assert sm.can_transition(TaskState.STOPPED)
        result = sm.transition(TaskState.STOPPED)
        assert result is True
        assert sm.state == TaskState.STOPPED

    def test_valid_transition_paused_to_stopped(self):
        """Test valid transition from PAUSED to STOPPED."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        sm.transition(TaskState.PAUSED)
        assert sm.can_transition(TaskState.STOPPED)
        result = sm.transition(TaskState.STOPPED)
        assert result is True
        assert sm.state == TaskState.STOPPED

    def test_invalid_transition_init_to_paused(self):
        """Test invalid transition from INIT to PAUSED."""
        sm = TaskStateMachine()
        assert not sm.can_transition(TaskState.PAUSED)
        result = sm.transition(TaskState.PAUSED)
        assert result is False
        assert sm.state == TaskState.INIT  # State unchanged

    def test_invalid_transition_stopped_to_running(self):
        """Test invalid transition from STOPPED to RUNNING."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        sm.transition(TaskState.STOPPED)
        assert not sm.can_transition(TaskState.RUNNING)
        result = sm.transition(TaskState.RUNNING)
        assert result is False
        assert sm.state == TaskState.STOPPED  # State unchanged

    def test_error_state_transition(self):
        """Test transition to ERROR state."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        result = sm.transition(TaskState.ERROR, "Test error message")
        assert result is True
        assert sm.state == TaskState.ERROR
        assert sm.error_message == "Test error message"

    def test_error_to_stopped(self):
        """Test transition from ERROR to STOPPED."""
        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)
        sm.transition(TaskState.ERROR)
        assert sm.can_transition(TaskState.STOPPED)
        result = sm.transition(TaskState.STOPPED)
        assert result is True
        assert sm.state == TaskState.STOPPED

    def test_force_state(self):
        """Test forcing state without validation."""
        sm = TaskStateMachine()
        sm.force_state(TaskState.STOPPED)
        assert sm.state == TaskState.STOPPED

    def test_is_running(self):
        """Test is_running helper method."""
        sm = TaskStateMachine()
        assert not sm.is_running()
        sm.transition(TaskState.RUNNING)
        assert sm.is_running()
        sm.transition(TaskState.PAUSED)
        assert not sm.is_running()

    def test_is_paused(self):
        """Test is_paused helper method."""
        sm = TaskStateMachine()
        assert not sm.is_paused()
        sm.transition(TaskState.RUNNING)
        assert not sm.is_paused()
        sm.transition(TaskState.PAUSED)
        assert sm.is_paused()

    def test_is_stopped(self):
        """Test is_stopped helper method."""
        sm = TaskStateMachine()
        assert not sm.is_stopped()
        sm.transition(TaskState.RUNNING)
        assert not sm.is_stopped()
        sm.transition(TaskState.STOPPED)
        assert sm.is_stopped()

    def test_thread_safety(self):
        """Test state machine thread safety."""
        import threading

        sm = TaskStateMachine()
        sm.transition(TaskState.RUNNING)

        errors = []

        def toggle_pause():
            try:
                for _ in range(100):
                    if sm.state == TaskState.RUNNING:
                        sm.transition(TaskState.PAUSED)
                    elif sm.state == TaskState.PAUSED:
                        sm.transition(TaskState.RUNNING)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_pause) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert sm.state in [TaskState.RUNNING, TaskState.PAUSED]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
