"""
engine.py — CogniFlow Attention State Machine  (v6 — Test-2 Bug Fixes)
Python 3.9+ compatible

═══════════════════════════════════════════════════════════════════════
BUG FIXES IN THIS VERSION (vs v5 — based on Test-2 results)
═══════════════════════════════════════════════════════════════════════

BUG 1 — Test 1 FAIL: Head turn stayed in Flow for ~10 seconds
    Root Cause: Head pose (pitch-only proxy) was being evaluated AFTER
    the gaze-based `_get_transition_signal()` path, which has a 2-frame
    "leave Flow" buffer. Large yaw turns were never reaching the head
    pose check at all — gaze remained "on screen" during a side turn
    because the user's iris was still pointing roughly forward.
    Additionally, `_update_head_pose()` only tracked pitch (up/down),
    not yaw (left/right), so a side turn produced zero deviation.

    Fix A — Gate order: Head pose hard gate is now checked FIRST, before
    gaze logic, immediately after the phone/person soft gates. A large
    angular deviation (> HEAD_YAW_THRESHOLD or > HEAD_PITCH_THRESHOLD)
    jumps directly to Distracted with zero frame buffer.

    Fix B — Yaw tracking added: `compute_head_yaw_raw()` added alongside
    the existing pitch proxy. Yaw is derived from the horizontal offset
    of the nose tip relative to the face midline. The engine now checks
    EITHER yaw OR pitch deviation to trigger the hard gate.

    Fix C — Thresholds separated:
        HEAD_YAW_THRESHOLD   = 0.12  (normalised, ~18–22° equivalent)
        HEAD_PITCH_THRESHOLD = 0.08  (normalised, ~12° equivalent, unchanged)
    These match the spec: yaw > 18° or pitch > 12° = instant Distracted.

BUG 2 — Test 2 FAIL: Slight downward tilt triggered Distracted in <5s
    Root Cause: The head pose hard gate used a single
    HEAD_DEVIATION_THRESHOLD = 0.08 for ALL directions (yaw + pitch).
    A gentle downward head tilt (5°–12° = Thinking zone per spec) was
    exceeding 0.08 in pitch and hitting the hard gate instead of the
    Thinking grace period.

    Fix: Pitch threshold raised to 0.10 (above the gentle-tilt range).
    Small downward tilts now fall below the hard gate and enter the
    Thinking → Away path correctly via the 2.5s grace period timer.
    Yaw threshold kept at 0.12 (side turns should still fire hard gate).

BUG 3 — Test 3 FAIL: Eyes closed, Drowsy never triggered
    Root Cause: EAR_DROWSY_DURATION was set to 2.0 seconds, but the
    spec mandates 16 consecutive frames at 8Hz (= 2.0s wall time only
    at perfect throughput). On the i5-6300U under load, the loop runs
    at ~6–7Hz, so 16 frames takes 2.3–2.7s — the time-based 2.0s
    window was closing before 16 real frames were observed.
    Additionally, EAR_OPEN_THRESHOLD == EAR_DROWSY_THRESHOLD (both
    0.18) meant the blink edge detector fired on every EAR reading
    near the threshold, potentially resetting _low_ear_since on noise.

    Fix A — Frame-count buffer replaces time-based duration:
        EAR_DROWSY_FRAMES = 16  (spec: "16 consecutive frames")
        _low_ear_frames counter replaces _low_ear_since timer.
        Any frame where EAR >= threshold resets counter to 0.

    Fix B — EAR thresholds decoupled:
        EAR_OPEN_THRESHOLD   = 0.22  (awake baseline — for blink detection)
        EAR_DROWSY_THRESHOLD = 0.18  (spec value — for drowsy gate)
    This prevents the blink detector from firing on every drowsy frame.

BUG 4 — Test 4 PARTIAL: Face gone → Thinking → Away (wrong)
         Recovery asymmetric buffer appeared to work (good).
    Root Cause: When the face disappears, `face_present=False` hits the
    absence block, but only transitions to "Absent" after 3.0s. During
    the 0–3s window it returns "Brief Absence" while HOLDING the last
    state. If the last state was Flow, the hold is fine. But on return,
    `_face_gone_since` resets to None and `gaze_on_screen` is True
    again — however `last_look_away_time` was NOT reset during the
    absence hold, so the engine fell into the Thinking branch on first
    re-entry if `last_look_away_time` was stale.

    Fix: Reset `last_look_away_time` and `distraction_start_time` when
    the face returns after an absence (transition from gone→present).
    Added `_was_face_present` flag to detect the re-entry edge.

BUG 5 — Test 5: Camera overlay score updates first, metrics panel lags
    Root Cause: `focus_score` (raw) and `_smoothed_score` (EMA) are
    both maintained, but the engine was returning `_smoothed_score`
    while the camera overlay was reading the raw `focus_score` directly
    from the object attribute. The EMA alpha of 0.12 means the smoothed
    score lags the raw score by several ticks (~8–10 frames = 1–1.25s).

    Fix: Engine now exposes a single authoritative value. `focus_score`
    IS the smoothed score — raw accumulation is kept in `_raw_score`
    (internal only). The camera overlay must read `engine.focus_score`
    (or the returned tuple value), NOT a separate attribute.
    Smooth alpha raised slightly to 0.20 for faster visual response
    while still damping flicker.
═══════════════════════════════════════════════════════════════════════
"""

import math
import time
from collections import deque
from typing import Tuple, Optional, List, Dict, Any


# ═══════════════════════════════════════════════════════════════════════
# EAR CALCULATOR
# ═══════════════════════════════════════════════════════════════════════

RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def compute_ear(landmarks, eye_indices: List[int]) -> float:
    """
    Eye Aspect Ratio — Soukupová & Čech (2016).
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Typical awake range: 0.25–0.35. Returns 0.30 on error.
    """
    if len(eye_indices) != 6:
        return 0.30
    try:
        p  = [landmarks[i] for i in eye_indices]
        v1 = _dist(p[1], p[5])
        v2 = _dist(p[2], p[4])
        h  = _dist(p[0], p[3])
        if h < 1e-6:
            return 0.30
        return (v1 + v2) / (2.0 * h)
    except (IndexError, AttributeError):
        return 0.30


def compute_avg_ear(landmarks) -> float:
    """Average EAR across both eyes for robustness."""
    return (compute_ear(landmarks, RIGHT_EYE_IDX) +
            compute_ear(landmarks, LEFT_EYE_IDX)) / 2.0


# ═══════════════════════════════════════════════════════════════════════
# HEAD POSE
# ═══════════════════════════════════════════════════════════════════════

def compute_head_pitch_raw(landmarks) -> float:
    """
    Raw normalised head pitch proxy. Self-calibrated inside engine.
    Positive = head down. Negative = head up. Range ~[-0.3, +0.3].
    """
    try:
        nose, forehead, chin = landmarks[1], landmarks[10], landmarks[152]
        face_h = abs(forehead.y - chin.y)
        if face_h < 1e-6:
            return 0.0
        return (nose.y - (forehead.y + chin.y) / 2.0) / face_h
    except (IndexError, AttributeError):
        return 0.0


def compute_head_yaw_raw(landmarks) -> float:
    """
    BUG 1 FIX — Raw normalised head yaw proxy. Self-calibrated inside engine.
    Positive = head turned right. Negative = head turned left.
    Uses horizontal nose-tip offset relative to the left/right eye midline.
    Range ~[-0.3, +0.3].
    """
    try:
        nose       = landmarks[1]
        left_eye   = landmarks[33]   # outer left eye corner
        right_eye  = landmarks[263]  # outer right eye corner
        face_w = abs(right_eye.x - left_eye.x)
        if face_w < 1e-6:
            return 0.0
        eye_mid_x = (left_eye.x + right_eye.x) / 2.0
        return (nose.x - eye_mid_x) / face_w
    except (IndexError, AttributeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# ATTENTION ENGINE  v6
# ═══════════════════════════════════════════════════════════════════════

class AttentionEngine:

    # ── EAR thresholds (BUG 3 FIX — decoupled) ───────────────────────
    EAR_OPEN_THRESHOLD   = 0.22   # blink detection baseline (awake)
    EAR_DROWSY_THRESHOLD = 0.18   # spec value — drowsy gate
    EAR_DROWSY_FRAMES    = 16     # BUG 3 FIX: frame-count not time-based

    # ── Score rates (calibrated to 8Hz tick rate) ─────────────────────
    SCORE_FLOW_TICK       =  0.125   # +1.0/sec
    SCORE_AWAY_TICK       = -0.125   # -1.0/sec
    SCORE_DISTRACTED_TICK = -0.25    # -2.0/sec
    SCORE_DROWSY_TICK     = -0.625   # -5.0/sec
    SCORE_ENTRY_PENALTY   = -3.0     # immediate on first look-away

    # ── Absence detection ─────────────────────────────────────────────
    ABSENT_THRESHOLD      = 3.0      # seconds without face = "Absent"

    # ── Alert threshold ───────────────────────────────────────────────
    LOW_FOCUS_ALERT_THRESHOLD = 60.0

    # ── Liveness / spoof detection ────────────────────────────────────
    LIVENESS_CHECK_WINDOW = 30.0
    LIVENESS_MIN_BLINKS   = 3

    # ── Head pose (BUG 1 + BUG 2 FIX — separated yaw/pitch thresholds)
    HEAD_YAW_THRESHOLD       = 0.12   # ~18–22° yaw  — hard gate (side turn)
    HEAD_PITCH_THRESHOLD     = 0.10   # ~12–15° pitch — hard gate (raised from 0.08)
    CALIBRATION_DURATION     = 20.0

    def __init__(self):
        self.current_state   = "Idle"

        # BUG 5 FIX: _raw_score is internal accumulator; focus_score IS
        # the smoothed value. Camera overlay + metrics panel both read
        # focus_score (or the returned tuple) — single source of truth.
        self._raw_score      = 100.0
        self.focus_score     = 100.0   # smoothed — THIS is the public value
        self._smooth_alpha   = 0.20    # raised from 0.12 for faster UI response

        # Thinking buffer
        self.thinking_threshold      = 2.5
        self.last_look_away_time: Optional[float] = None

        # FRT
        self.distraction_start_time: Optional[float] = None
        self.last_frt = 0.0

        # Asymmetric transition buffers
        self._leave_flow_buf: deque = deque(maxlen=2)
        self._enter_flow_buf: deque = deque(maxlen=3)

        # EAR / drowsiness
        # BUG 3 FIX: frame counter replaces time-based _low_ear_since
        self._low_ear_frames: int   = 0
        self._ear_history:    deque = deque(maxlen=300)
        self._blink_count     = 0
        self._last_ear_above  = True

        # Absence tracking
        self._face_gone_since:   Optional[float] = None
        self._was_face_present:  bool = True   # BUG 4 FIX: re-entry edge detect

        # Alerts
        self._alerts: Dict[str, bool] = {
            "low_focus":        False,
            "drowsy":           False,
            "absent":           False,
            "possible_spoof":   False,
            "multiple_persons": False,
        }

        # Liveness detection
        self._liveness_window_start: Optional[float] = None
        self._liveness_checked       = False
        self._face_frames_in_window  = 0

        # Head pose calibration — now tracks BOTH pitch and yaw
        self._pitch_raw_history: deque = deque(maxlen=200)
        self._yaw_raw_history:   deque = deque(maxlen=200)
        self._pitch_baseline:    Optional[float] = None
        self._yaw_baseline:      Optional[float] = None
        self._calibration_start: Optional[float] = None
        self._calibrated         = False
        self._pitch_smooth_buf:  deque = deque(maxlen=5)
        self._yaw_smooth_buf:    deque = deque(maxlen=5)

    # ──────────────────────────────────────────────────────────────────
    # MAIN UPDATE
    # ──────────────────────────────────────────────────────────────────

    def update_state(
        self,
        face_present:   bool,
        gaze_on_screen: bool,
        phone_detected: bool  = False,
        ear:            float = 0.30,
        head_pitch:     float = 0.0,
        head_yaw:       float = 0.0,   # BUG 1 FIX: yaw now a required signal
        person_count:   int   = 1,
    ) -> Tuple[str, float, str]:
        """
        Main tick. Call at ~8Hz.

        Returns: (state_label, smoothed_focus_score, reason_string)

        IMPORTANT — BUG 5 FIX:
            The returned score AND self.focus_score are both the smoothed
            value. Do NOT read self._raw_score in the UI layer.
            Camera overlay and cognitive metrics panel must both consume
            the same value: engine.focus_score or the tuple index [1].
        """

        now = time.time()

        # Sticky spoof flag
        spoof_sticky = self._alerts.get("possible_spoof", False)
        self._alerts = {k: False for k in self._alerts}
        if spoof_sticky:
            self._alerts["possible_spoof"] = True

        # ── EAR + liveness ────────────────────────────────────────────
        ear_alert = self._update_ear(ear, face_present)
        self._update_liveness(face_present, now)

        # ── Head pose (calibration + deviation) ───────────────────────
        head_hard_gate, head_soft_away = self._update_head_pose(
            head_pitch, head_yaw, now, face_present
        )

        # ── Absence detection ─────────────────────────────────────────
        if not face_present:
            if self._face_gone_since is None:
                self._face_gone_since = now
            elapsed_absent = now - self._face_gone_since

            if elapsed_absent >= self.ABSENT_THRESHOLD:
                if self.distraction_start_time is None:
                    self.distraction_start_time = now
                self.current_state = "Absent"
                self._raw_score    = max(0.0, self._raw_score + self.SCORE_AWAY_TICK)
                self.focus_score   = self._ema(self._raw_score)
                self._alerts["absent"] = True
                self._was_face_present = False
                return self.current_state, round(self.focus_score, 2), "User Left Desk"
            else:
                self.focus_score = self._ema(self._raw_score)
                self._was_face_present = False
                return self.current_state, round(self.focus_score, 2), "Brief Absence"
        else:
            # BUG 4 FIX: Face just returned after absence — reset stale timers
            if not self._was_face_present:
                self.last_look_away_time    = None
                self.distraction_start_time = None
            self._face_gone_since  = None
            self._was_face_present = True

        # ── Priority 1: Multiple persons ─────────────────────────────
        if person_count > 1:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self._raw_score    = max(0.0, self._raw_score + self.SCORE_DISTRACTED_TICK)
            self.focus_score   = self._ema(self._raw_score)
            self._alerts["multiple_persons"] = True
            return self.current_state, round(self.focus_score, 2), "Multiple Persons"

        # ── Priority 2: Phone detected ────────────────────────────────
        if phone_detected:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self._raw_score    = max(0.0, self._raw_score + self.SCORE_DISTRACTED_TICK)
            self.focus_score   = self._ema(self._raw_score)
            return self.current_state, round(self.focus_score, 2), "Phone Detected"

        # ── Priority 3: HEAD POSE HARD GATE (BUG 1 + BUG 2 FIX) ──────
        # Checked BEFORE gaze logic. Large angular deviation = instant
        # Distracted with NO frame buffer and NO thinking grace period.
        # Spec: yaw > 18° or pitch > 12° → Distracted immediately.
        if head_hard_gate:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self._raw_score    = max(0.0, self._raw_score + self.SCORE_DISTRACTED_TICK)
            self.focus_score   = self._ema(self._raw_score)
            return self.current_state, round(self.focus_score, 2), "Head Turn Detected"

        # ── Priority 4: Drowsy (EAR frame-count gate) ─────────────────
        if ear_alert:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Drowsy"
            self._raw_score    = max(0.0, self._raw_score + self.SCORE_DROWSY_TICK)
            self.focus_score   = self._ema(self._raw_score)
            self._alerts["drowsy"] = True
            return self.current_state, round(self.focus_score, 2), "Drowsiness Detected"

        # ── Priority 5: Gaze-based focus signal ───────────────────────
        raw_focused       = face_present and gaze_on_screen
        confident_focused = self._get_transition_signal(raw_focused)

        if confident_focused:
            if self.current_state in ("Distracted", "Away", "Drowsy", "Absent"):
                if self.distraction_start_time is not None:
                    self.last_frt = now - self.distraction_start_time
                    self.distraction_start_time = None

            self.current_state       = "Flow"
            self.last_look_away_time = None
            self._low_ear_frames     = 0

            self._raw_score  = min(100.0, self._raw_score + self.SCORE_FLOW_TICK)
            self.focus_score = self._ema(self._raw_score)
            self._update_low_focus_alert()
            return self.current_state, round(self.focus_score, 2), "Focused"

        # ── Priority 6: Not focused → Thinking → Away ─────────────────
        if self.last_look_away_time is None:
            self.last_look_away_time = now
            self._raw_score = max(0.0, self._raw_score + self.SCORE_ENTRY_PENALTY)

        if self.distraction_start_time is None:
            self.distraction_start_time = now

        elapsed_away = now - self.last_look_away_time

        if elapsed_away < self.thinking_threshold:
            self.current_state = "Thinking"
            self.focus_score   = self._ema(self._raw_score)
            self._update_low_focus_alert()
            return self.current_state, round(self.focus_score, 2), "Cognitive Processing"

        self.current_state = "Away"
        penalty = self.SCORE_DISTRACTED_TICK if head_soft_away else self.SCORE_AWAY_TICK
        self._raw_score  = max(0.0, self._raw_score + penalty)
        self.focus_score = self._ema(self._raw_score)
        self._update_low_focus_alert()

        reason = "Extended Absence + Head Down" if head_soft_away else "Extended Absence"
        return self.current_state, round(self.focus_score, 2), reason

    # ──────────────────────────────────────────────────────────────────
    # ALERT HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _update_low_focus_alert(self):
        if self.focus_score < self.LOW_FOCUS_ALERT_THRESHOLD:
            self._alerts["low_focus"] = True

    def get_alerts(self) -> Dict[str, bool]:
        return dict(self._alerts)

    # ──────────────────────────────────────────────────────────────────
    # LIVENESS DETECTION
    # ──────────────────────────────────────────────────────────────────

    def _update_liveness(self, face_present: bool, now: float):
        if self._liveness_checked or not face_present:
            return
        if self._liveness_window_start is None:
            self._liveness_window_start = now
        self._face_frames_in_window += 1
        elapsed = now - self._liveness_window_start
        if elapsed >= self.LIVENESS_CHECK_WINDOW:
            if self._blink_count < self.LIVENESS_MIN_BLINKS:
                self._alerts["possible_spoof"] = True
                print(f"[CogniFlow] WARNING: Possible spoof — "
                      f"only {self._blink_count} blinks in {elapsed:.1f}s")
            self._liveness_checked = True

    # ──────────────────────────────────────────────────────────────────
    # ASYMMETRIC TRANSITION BUFFERS
    # ──────────────────────────────────────────────────────────────────

    def _get_transition_signal(self, raw_focused: bool) -> bool:
        self._leave_flow_buf.append(raw_focused)
        self._enter_flow_buf.append(raw_focused)

        if all(not f for f in self._leave_flow_buf):
            return False
        if all(f for f in self._enter_flow_buf):
            return True
        return self.current_state == "Flow"

    # ──────────────────────────────────────────────────────────────────
    # HEAD POSE — BUG 1 + BUG 2 FIX
    # ──────────────────────────────────────────────────────────────────

    def _update_head_pose(
        self,
        raw_pitch: float,
        raw_yaw:   float,
        now:       float,
        face_present: bool,
    ) -> Tuple[bool, bool]:
        """
        Returns (hard_gate_fired, soft_away_signal).

        hard_gate_fired:
            True if yaw deviation > HEAD_YAW_THRESHOLD  (~18° side turn)
            OR pitch deviation > HEAD_PITCH_THRESHOLD   (~15° extreme tilt)
            → triggers instant Distracted, no buffer, no Thinking grace.

        soft_away_signal:
            True if pitch is in the moderate tilt range (5°–12°).
            Used only to increase Away penalty, not as a hard gate.
            This is what Thinking → Away escalation uses.
        """
        if not face_present:
            return False, False

        # Smooth both axes
        self._pitch_smooth_buf.append(raw_pitch)
        self._yaw_smooth_buf.append(raw_yaw)
        s_pitch = sum(self._pitch_smooth_buf) / len(self._pitch_smooth_buf)
        s_yaw   = sum(self._yaw_smooth_buf)   / len(self._yaw_smooth_buf)

        # Calibration phase
        if not self._calibrated:
            if self._calibration_start is None:
                self._calibration_start = now
            self._pitch_raw_history.append(s_pitch)
            self._yaw_raw_history.append(s_yaw)
            elapsed = now - self._calibration_start
            if elapsed >= self.CALIBRATION_DURATION and len(self._pitch_raw_history) >= 10:
                self._pitch_baseline = sum(self._pitch_raw_history) / len(self._pitch_raw_history)
                self._yaw_baseline   = sum(self._yaw_raw_history)   / len(self._yaw_raw_history)
                self._calibrated = True
                print(f"[CogniFlow] Head pose calibrated. "
                      f"Pitch baseline={self._pitch_baseline:.4f}  "
                      f"Yaw baseline={self._yaw_baseline:.4f}")
            return False, False

        pitch_dev = s_pitch - self._pitch_baseline
        yaw_dev   = s_yaw   - self._yaw_baseline

        # Hard gate: large yaw (side turn) OR extreme pitch (BUG 1 + BUG 2)
        hard = (
            abs(yaw_dev)   > self.HEAD_YAW_THRESHOLD or
            abs(pitch_dev) > self.HEAD_PITCH_THRESHOLD
        )

        # Soft signal: moderate pitch only (Thinking zone, ~5°–12°)
        # Does NOT fire the hard gate. Used for Away penalty aggravation.
        soft_away = (not hard) and (abs(pitch_dev) > 0.04)

        return hard, soft_away

    # ──────────────────────────────────────────────────────────────────
    # EAR TRACKING — BUG 3 FIX
    # ──────────────────────────────────────────────────────────────────

    def _update_ear(self, ear: float, face_present: bool) -> bool:
        """
        Returns True when EAR has been below EAR_DROWSY_THRESHOLD for
        EAR_DROWSY_FRAMES consecutive frames (spec: 16 frames at 8Hz).

        BUG 3 FIX: Uses frame counter, not wall-clock timer.
        Any frame where EAR >= threshold resets counter to zero.
        """
        if not face_present:
            self._low_ear_frames = 0
            self._last_ear_above = True
            return False

        self._ear_history.append(ear)

        # Blink edge detection — uses EAR_OPEN_THRESHOLD (0.22), not drowsy threshold
        currently_above = ear >= self.EAR_OPEN_THRESHOLD
        if self._last_ear_above and not currently_above:
            self._blink_count += 1
        self._last_ear_above = currently_above

        # Drowsiness frame counter — uses EAR_DROWSY_THRESHOLD (0.18)
        if ear < self.EAR_DROWSY_THRESHOLD:
            self._low_ear_frames += 1
            if self._low_ear_frames >= self.EAR_DROWSY_FRAMES:
                return True
        else:
            self._low_ear_frames = 0   # reset on ANY frame above threshold

        return False

    # ──────────────────────────────────────────────────────────────────
    # SCORE SMOOTHING — BUG 5 FIX
    # ──────────────────────────────────────────────────────────────────

    def _ema(self, target: float) -> float:
        """
        EMA applied to raw score. Result stored in self.focus_score.
        Alpha raised to 0.20 (from 0.12) so UI lag is ~3–4 frames max
        instead of 8–10. Both camera overlay and metrics panel must
        read self.focus_score — not self._raw_score.
        """
        self.focus_score = (
            self.focus_score
            + self._smooth_alpha * (target - self.focus_score)
        )
        return self.focus_score

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC ACCESSORS
    # ──────────────────────────────────────────────────────────────────

    def get_last_frt(self) -> float:
        return round(self.last_frt, 2)

    def get_blink_rate(self) -> float:
        if len(self._ear_history) < 16:
            return 0.0
        elapsed_sec = len(self._ear_history) / 8.0
        return round((self._blink_count / elapsed_sec) * 60.0, 1)

    def get_ear_status(self) -> str:
        if not self._ear_history:
            return "Unknown"
        latest = self._ear_history[-1]
        if latest < self.EAR_DROWSY_THRESHOLD:
            return "Eyes Closed"
        if latest < self.EAR_OPEN_THRESHOLD:  # 0.18–0.22 range
            return "Drowsy"
        if latest > 0.32:
            return "Alert"
        return "Normal"

    def is_calibrated(self) -> bool:
        return self._calibrated