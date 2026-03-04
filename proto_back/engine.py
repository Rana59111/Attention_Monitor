"""
engine.py — CogniFlow Attention State Machine
Compatible with Python 3.9+

PY39: tuple[str, float, str] return hint replaced with Tuple from typing
"""

import time
from typing import Tuple


class AttentionEngine:
    def __init__(self):
        self.current_state = "Idle"
        self.focus_score = 100.0

        # Temporal Buffer — 4 s grace period before a look-away counts as distraction
        self.thinking_threshold = 4.0
        self.last_look_away_time = None

        # Focus Recovery Time tracking
        self.distraction_start_time = None
        self.last_frt = 0.0

    def update_state(
        self,
        face_present: bool,
        gaze_on_screen: bool,
        phone_detected: bool = False,
    ) -> Tuple[str, float, str]:          # ← Python 3.9 compatible (was tuple[...])
        """
        Process one frame's worth of AI signals and return the new state.

        Returns:
            (state, focus_score, reason)
            state       — "Flow" | "Thinking" | "Away" | "Distracted" | "Idle"
            focus_score — float 0.0–100.0
            reason      — short string used by CogniBot to generate nudges
        """
        now = time.time()

        # ── Priority 1: phone beats everything ────────────────────────────────
        if phone_detected:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self.focus_score = max(0.0, self.focus_score - 2.0)
            return self.current_state, self.focus_score, "Phone Detected"

        # ── Priority 2: face present AND looking at screen → Flow ─────────────
        if face_present and gaze_on_screen:
            # Coming back from distraction — record FRT
            if self.current_state in ("Distracted", "Away"):
                if self.distraction_start_time is not None:
                    self.last_frt = now - self.distraction_start_time
                    self.distraction_start_time = None

            self.current_state = "Flow"   # was "Flow State" — fixed to match frontend
            self.focus_score = min(100.0, self.focus_score + 1.0)
            self.last_look_away_time = None
            return self.current_state, self.focus_score, "Focused"

        # ── Priority 3: look-away / no face → Thinking buffer → Away ──────────
        if not gaze_on_screen or not face_present:
            if self.last_look_away_time is None:
                self.last_look_away_time = now
            if self.distraction_start_time is None:
                self.distraction_start_time = now

            elapsed_away = now - self.last_look_away_time

            if elapsed_away < self.thinking_threshold:
                # Still within the 4-second grace period
                self.current_state = "Thinking"
                # Score stays the same during Thinking
                return self.current_state, self.focus_score, "Cognitive Processing"
            else:
                # Grace period expired — mark as Away and start penalising
                self.current_state = "Away"
                self.focus_score = max(0.0, self.focus_score - 1.0)
                return self.current_state, self.focus_score, "Extended Absence"

        # ── Fallback (should not normally be reached) ─────────────────────────
        return "Idle", self.focus_score, "Idle"

    def get_last_frt(self) -> float:
        """Return the most recent Focus Recovery Time in seconds (rounded to 2 dp)."""
        return round(self.last_frt, 2)