
"""Deterministic Safety-only v0 intervention.

No CBS calls. Keeps the IL policy unchanged.
"""

from __future__ import annotations


class SafetyOnlyV0:
    def __init__(self, hold_steps: int = 2):
        if hold_steps < 1:
            raise ValueError("hold_steps must be >= 1")
        self.hold_steps = hold_steps
        self.reset()

    def reset(self):
        self.selected_agent = None
        self.remaining = 0
        self.triggered = False

    def arm(self, positions, goals):
        """Select the lowest-ID active agent once per episode."""
        if self.triggered:
            return False

        active = [
            aid for aid in sorted(positions)
            if positions[aid] != goals[aid]
        ]
        if not active:
            return False

        self.selected_agent = active[0]
        self.remaining = self.hold_steps
        self.triggered = True
        return True

    def apply(self, actions):
        """Override only the selected agent with WAIT."""
        result = dict(actions)

        if self.remaining > 0:
            result[self.selected_agent] = 4
            self.remaining -= 1

        return result
