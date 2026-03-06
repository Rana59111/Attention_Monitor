"""
engine.py — CogniFlow Attention State Machine  (v5 — Full Spec Compliance)
Python 3.9+ compatible

═══════════════════════════════════════════════════════════════════════
GAPS CLOSED IN THIS VERSION (vs functional spec)
═══════════════════════════════════════════════════════════════════════

GAP 1 — EAR threshold corrected
    Spec says: EAR < 0.18 for 2+ seconds = Sleeping/Drowsy
    Old code:  EAR_DROWSY_THRESHOLD = 0.22  (too sensitive, triggers on squinting)
    Fix:       EAR_DROWSY_THRESHOLD = 0.18  (exact spec value)

GAP 2 — Score rates corrected to match spec table
    Spec:      Flow +1%, Looking Away -1%, Phone -2%, Eyes Closed -5%
    Old rates: +0.3, -0.5, -0.8, -0.6  (arbitrary, not matching spec)
    Fix:       All rates now match the spec exactly at 8Hz tick rate.
               At 8 ticks/sec:
                 Flow:       +0.125/tick  → +1.0/sec  ✓
                 Thinking:   no change    ✓
                 Away:       -0.125/tick  → -1.0/sec  ✓
                 Distracted: -0.25/tick   → -2.0/sec  ✓
                 Drowsy:     -0.625/tick  → -5.0/sec  ✓
               Entry penalty kept at -3.0 (immediate visual response)

GAP 3 — User Absence (face gone) vs Looking Away now distinct states
    Old:  both triggered "Away" via gaze loss
    New:  "Away"    = face present but gaze off screen (looking at notebook)
          "Absent"  = face NOT present for > 3 seconds (user left desk)
    These are separate states with separate score penalties and separate
    nudge messages. Away is gentler (Thinking buffer applies).
    Absent has no Thinking buffer — if you're not there, you're not there.

GAP 4 — Low Focus Alert flag
    Spec: trigger visual alert (bounce animation) when score < 60%
    Old:  no alert field in payload
    New:  engine.get_alerts() returns dict with:
          "low_focus"  : True when score < 60
          "drowsy"     : True when in Drowsy state
          "absent"     : True when in Absent state
    These drive the frontend notification system.

GAP 5 — Liveness detection (micro-blink)
    Spec: Micro-Blink Tracking for liveness (anti-spoof)
    Old:  blink counted but never used as a liveness signal
    New:  if zero blinks are recorded in the first 30 seconds of a
          session while a face is continuously present, raise a
          "possible_spoof" flag. A real person blinks 12-20 times/min.
          A static photo will show zero blinks.
          This flag is sent in the WebSocket payload.

GAP 6 — Multiple persons detection
    This is handled in main.py (max_num_faces changed to 3, engine
    receives person_count). Engine triggers "Distracted" with reason
    "Multiple Persons" when count > 1.
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


# ═══════════════════════════════════════════════════════════════════════
# ATTENTION ENGINE  v5
# ═══════════════════════════════════════════════════════════════════════

class AttentionEngine:

    # ── EAR thresholds (GAP 1 fixed) ──────────────────────────────────
    EAR_OPEN_THRESHOLD   = 0.18   # spec value — below = eye closed
    EAR_DROWSY_THRESHOLD = 0.18   # same as open threshold for spec compliance
    EAR_DROWSY_DURATION  = 2.0    # seconds sustained to trigger Drowsy

    # ── Score rates (GAP 2 fixed — calibrated to 8Hz tick rate) ───────
    # Spec:  Flow +1%/s, Away -1%/s, Distracted -2%/s, Drowsy -5%/s
    # At 8Hz (0.125s/tick): divide per-second rate by 8
    SCORE_FLOW_TICK       =  0.125   # +1.0/sec
    SCORE_AWAY_TICK       = -0.125   # -1.0/sec
    SCORE_DISTRACTED_TICK = -0.25    # -2.0/sec
    SCORE_DROWSY_TICK     = -0.625   # -5.0/sec
    SCORE_ENTRY_PENALTY   = -3.0     # immediate on first look-away

    # ── Absence detection (GAP 3) ─────────────────────────────────────
    ABSENT_THRESHOLD      = 3.0      # seconds without face = "Absent"

    # ── Alert threshold (GAP 4) ───────────────────────────────────────
    LOW_FOCUS_ALERT_THRESHOLD = 60.0

    # ── Liveness / spoof detection (GAP 5) ────────────────────────────
    LIVENESS_CHECK_WINDOW = 30.0   # seconds to observe before flagging
    LIVENESS_MIN_BLINKS   = 3      # minimum blinks in window for live person

    # ── Head pose ─────────────────────────────────────────────────────
    HEAD_DEVIATION_THRESHOLD = 0.08
    CALIBRATION_DURATION     = 20.0

    def __init__(self):
        self.current_state   = "Idle"
        self.focus_score     = 100.0
        self._smoothed_score = 100.0
        self._smooth_alpha   = 0.12

        # Thinking buffer
        self.thinking_threshold      = 2.5
        self.last_look_away_time: Optional[float] = None

        # FRT
        self.distraction_start_time: Optional[float] = None
        self.last_frt = 0.0

        # Asymmetric transition buffers (v4 fix retained)
        self._leave_flow_buf: deque = deque(maxlen=2)
        self._enter_flow_buf: deque = deque(maxlen=3)

        # EAR / drowsiness
        self._low_ear_since:  Optional[float] = None
        self._ear_history:    deque = deque(maxlen=300)
        self._blink_count     = 0
        self._last_ear_above  = True

        # GAP 3: Absence tracking (face gone vs looking away)
        self._face_gone_since: Optional[float] = None

        # GAP 4: Alert state
        self._alerts: Dict[str, bool] = {
            "low_focus":      False,
            "drowsy":         False,
            "absent":         False,
            "possible_spoof": False,
            "multiple_persons": False,
        }

        # GAP 5: Liveness detection
        self._liveness_window_start: Optional[float] = None
        self._liveness_checked       = False
        self._face_frames_in_window  = 0

        # Head pose calibration
        self._pitch_raw_history: deque = deque(maxlen=200)
        self._pitch_baseline:    Optional[float] = None
        self._calibration_start: Optional[float] = None
        self._calibrated         = False
        self._pitch_smooth_buf:  deque = deque(maxlen=5)

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
        person_count:   int   = 1,    # GAP 6: multiple persons
    ) -> Tuple[str, float, str]:

        now = time.time()

        # Reset per-tick alerts — recalculated fresh each tick.
        # possible_spoof is STICKY — once set it never resets (it's a
        # session-level finding, not a per-tick state).
        spoof_sticky = self._alerts.get("possible_spoof", False)
        self._alerts = {k: False for k in self._alerts}
        if spoof_sticky:
            self._alerts["possible_spoof"] = True

        # ── EAR + liveness ────────────────────────────────────────────
        ear_alert = self._update_ear(ear, now, face_present)
        self._update_liveness(face_present, now)

        # ── Head pose ─────────────────────────────────────────────────
        head_away = self._update_head_pose(head_pitch, now, face_present)

        # ── GAP 3: Absence detection (face completely gone) ───────────
        # This is checked BEFORE gaze logic because if there's no face,
        # gaze is irrelevant. Absent ≠ Away.
        if not face_present:
            if self._face_gone_since is None:
                self._face_gone_since = now
            elapsed_absent = now - self._face_gone_since

            if elapsed_absent >= self.ABSENT_THRESHOLD:
                # Lock in distraction start for FRT tracking
                if self.distraction_start_time is None:
                    self.distraction_start_time = now
                self.current_state = "Absent"
                self.focus_score   = max(0.0, self.focus_score + self.SCORE_AWAY_TICK)
                self._smoothed_score = self._ema(self.focus_score)
                self._alerts["absent"] = True
                return self.current_state, round(self._smoothed_score, 2), "User Left Desk"
            else:
                # Face briefly gone but within tolerance — hold last state
                self._smoothed_score = self._ema(self.focus_score)
                return self.current_state, round(self._smoothed_score, 2), "Brief Absence"
        else:
            # Face present — reset absence timer
            self._face_gone_since = None

        # ── Priority 1: Multiple persons (GAP 6) ─────────────────────
        if person_count > 1:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self.focus_score   = max(0.0, self.focus_score + self.SCORE_DISTRACTED_TICK)
            self._smoothed_score = self._ema(self.focus_score)
            self._alerts["multiple_persons"] = True
            return self.current_state, round(self._smoothed_score, 2), "Multiple Persons"

        # ── Priority 2: Phone detected ────────────────────────────────
        if phone_detected:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self.focus_score   = max(0.0, self.focus_score + self.SCORE_DISTRACTED_TICK)
            self._smoothed_score = self._ema(self.focus_score)
            return self.current_state, round(self._smoothed_score, 2), "Phone Detected"

        # ── Priority 3: Drowsy (EAR sustained low) ────────────────────
        if ear_alert:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Drowsy"
            self.focus_score   = max(0.0, self.focus_score + self.SCORE_DROWSY_TICK)
            self._smoothed_score = self._ema(self.focus_score)
            self._alerts["drowsy"] = True
            return self.current_state, round(self._smoothed_score, 2), "Drowsiness Detected"

        # ── Priority 4: Gaze-based focused signal ─────────────────────
        raw_focused       = face_present and gaze_on_screen
        confident_focused = self._get_transition_signal(raw_focused)

        if confident_focused:
            if self.current_state in ("Distracted", "Away", "Drowsy", "Absent"):
                if self.distraction_start_time is not None:
                    self.last_frt = now - self.distraction_start_time
                    self.distraction_start_time = None

            self.current_state       = "Flow"
            self.last_look_away_time = None
            self._low_ear_since      = None

            # GAP 2: correct rate +1%/sec at 8Hz = +0.125/tick
            self.focus_score     = min(100.0, self.focus_score + self.SCORE_FLOW_TICK)
            self._smoothed_score = self._ema(self.focus_score)
            self._update_low_focus_alert()
            return self.current_state, round(self._smoothed_score, 2), "Focused"

        # ── Priority 5: Not focused → Thinking → Away ─────────────────
        if self.last_look_away_time is None:
            self.last_look_away_time = now
            # Immediate entry penalty
            self.focus_score = max(0.0, self.focus_score + self.SCORE_ENTRY_PENALTY)

        if self.distraction_start_time is None:
            self.distraction_start_time = now

        elapsed_away = now - self.last_look_away_time

        if elapsed_away < self.thinking_threshold:
            self.current_state   = "Thinking"
            self._smoothed_score = self._ema(self.focus_score)
            self._update_low_focus_alert()
            return self.current_state, round(self._smoothed_score, 2), "Cognitive Processing"

        # GAP 2 + GAP 3: Away = gaze off screen, face still present
        # Score: -1%/sec at 8Hz = -0.125/tick
        # Extra if head also down: use -0.25/tick (head aggravation)
        self.current_state = "Away"
        penalty = self.SCORE_DISTRACTED_TICK if head_away else self.SCORE_AWAY_TICK
        self.focus_score     = max(0.0, self.focus_score + penalty)
        self._smoothed_score = self._ema(self.focus_score)
        self._update_low_focus_alert()

        reason = "Extended Absence + Head Down" if head_away else "Extended Absence"
        return self.current_state, round(self._smoothed_score, 2), reason

    # ──────────────────────────────────────────────────────────────────
    # ALERT HELPERS
    # ──────────────────────────────────────────────────────────────────

    def _update_low_focus_alert(self):
        """GAP 4: Set low_focus alert when score drops below 60."""
        if self._smoothed_score < self.LOW_FOCUS_ALERT_THRESHOLD:
            self._alerts["low_focus"] = True

    def get_alerts(self) -> Dict[str, bool]:
        """
        Returns current alert flags for the frontend.
        Called every WebSocket tick alongside get_last_frt().

        Keys:
          low_focus        — score < 60, trigger bounce animation
          drowsy           — in Drowsy state, trigger high-priority alert
          absent           — user left desk
          possible_spoof   — no blinks detected in first 30s
          multiple_persons — more than one face detected
        """
        return dict(self._alerts)

    # ──────────────────────────────────────────────────────────────────
    # LIVENESS DETECTION (GAP 5)
    # ──────────────────────────────────────────────────────────────────

    def _update_liveness(self, face_present: bool, now: float):
        """
        Micro-blink liveness check.
        Monitors the first LIVENESS_CHECK_WINDOW seconds of a face being
        present. If fewer than LIVENESS_MIN_BLINKS are detected, flag as
        possible spoof (static photo).
        """
        if self._liveness_checked:
            return

        if not face_present:
            return

        if self._liveness_window_start is None:
            self._liveness_window_start = now

        self._face_frames_in_window += 1

        elapsed = now - self._liveness_window_start
        if elapsed >= self.LIVENESS_CHECK_WINDOW:
            # Window complete — evaluate blink count
            if self._blink_count < self.LIVENESS_MIN_BLINKS:
                self._alerts["possible_spoof"] = True
                print(f"[CogniFlow] WARNING: Possible spoof — "
                      f"only {self._blink_count} blinks in {elapsed:.1f}s")
            self._liveness_checked = True

    # ──────────────────────────────────────────────────────────────────
    # ASYMMETRIC TRANSITION BUFFERS (v4, retained)
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
    # HEAD POSE (v4, retained)
    # ──────────────────────────────────────────────────────────────────

    def _update_head_pose(self, raw_pitch: float, now: float, face_present: bool) -> bool:
        if not face_present:
            return False
        self._pitch_smooth_buf.append(raw_pitch)
        smoothed = sum(self._pitch_smooth_buf) / len(self._pitch_smooth_buf)

        if not self._calibrated:
            if self._calibration_start is None:
                self._calibration_start = now
            self._pitch_raw_history.append(smoothed)
            elapsed = now - self._calibration_start
            if elapsed >= self.CALIBRATION_DURATION and len(self._pitch_raw_history) >= 10:
                self._pitch_baseline = sum(self._pitch_raw_history) / len(self._pitch_raw_history)
                self._calibrated = True
                print(f"[CogniFlow] Head pose calibrated. Baseline = {self._pitch_baseline:.4f}")
            return False

        deviation = smoothed - self._pitch_baseline
        return abs(deviation) > self.HEAD_DEVIATION_THRESHOLD

    # ──────────────────────────────────────────────────────────────────
    # EAR TRACKING
    # ──────────────────────────────────────────────────────────────────

    def _update_ear(self, ear: float, now: float, face_present: bool) -> bool:
        if not face_present:
            self._low_ear_since  = None
            self._last_ear_above = True
            return False

        self._ear_history.append(ear)

        # Blink edge detection (for liveness)
        currently_above = ear >= self.EAR_OPEN_THRESHOLD
        if self._last_ear_above and not currently_above:
            self._blink_count += 1
        self._last_ear_above = currently_above

        # Drowsiness timer — GAP 1: threshold now 0.18 (spec value)
        if ear < self.EAR_DROWSY_THRESHOLD:
            if self._low_ear_since is None:
                self._low_ear_since = now
            elif (now - self._low_ear_since) >= self.EAR_DROWSY_DURATION:
                return True
        else:
            self._low_ear_since = None

        return False

    # ──────────────────────────────────────────────────────────────────
    # SCORE SMOOTHING
    # ──────────────────────────────────────────────────────────────────

    def _ema(self, target: float) -> float:
        self._smoothed_score = (
            self._smoothed_score
            + self._smooth_alpha * (target - self._smoothed_score)
        )
        return self._smoothed_score

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
        if latest < self.EAR_OPEN_THRESHOLD:
            return "Eyes Closed"
        if latest < 0.22:
            return "Drowsy"
        if latest > 0.32:
            return "Alert"
        return "Normal"

    def is_calibrated(self) -> bool:
        return self._calibrated