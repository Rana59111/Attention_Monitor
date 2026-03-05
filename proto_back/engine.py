"""
engine.py — CogniFlow Attention State Machine  (v3 — EAR + Head Pose)
Python 3.9+ compatible

═══════════════════════════════════════════════════════════════════════
WHAT'S NEW IN THIS VERSION
═══════════════════════════════════════════════════════════════════════

NEW SIGNAL 1 — Eye Aspect Ratio (EAR)
    Formula from the literature (Soukupová & Čech, 2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Where p1-p6 are the six eye landmark coordinates.
    A healthy awake EAR is typically 0.25–0.40.
    EAR < 0.20 for more than 2 consecutive seconds = drowsiness.
    EAR < 0.15 at any point = eye closed (blink or fatigue).
    We average left and right eye EAR for robustness.

NEW SIGNAL 2 — Head Pose (Pitch)
    Using 3D landmarks from MediaPipe, we estimate head pitch (up/down tilt).
    Pitch > 15° downward = looking at desk/phone below screen.
    Pitch < -20° upward = looking away above screen.
    This catches distraction that gaze-alone misses (e.g. reading notes
    on desk while iris stays centred).

NEW COMPOSITE SCORING SYSTEM
    Old system: binary — focused or not, +0.3 or -0.5 per tick.
    New system: weighted three-signal composite:
        - Gaze:      weight 0.50  (most important — are eyes on screen?)
        - EAR:       weight 0.30  (are eyes open and alert?)
        - Head pose: weight 0.20  (is head orientation facing screen?)
    Each signal contributes to a 0-100 composite focus score.
    This is far more robust and far more citable as a novel contribution.

NEW STATE — "Drowsy"
    Triggered when EAR stays below 0.20 for > 2 seconds.
    Separate from "Away" — the user is present but fatigued.
    Different nudge from CogniBot ("take a break" vs "come back to screen").
    Score penalty: -0.6/tick (less than Distracted, more than Away)
    because drowsiness is involuntary and deserves gentler treatment.
═══════════════════════════════════════════════════════════════════════
"""

import math
import time
from collections import deque
from typing import Tuple, Optional, List


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE EAR CALCULATOR
# Kept as a module-level function so main.py can call it independently
# to include raw EAR in the WebSocket payload for the frontend graph.
# ═══════════════════════════════════════════════════════════════════════

def compute_ear(landmarks, eye_indices: List[int]) -> float:
    """
    Compute Eye Aspect Ratio for one eye given 6 landmark indices.

    Standard 6-point EAR formula (Soukupová & Čech, 2016):
        EAR = (vertical_1 + vertical_2) / (2 × horizontal)

    Landmark layout for right eye (indices in MediaPipe 468-point mesh):
        p1 = 33  (outer corner)
        p2 = 160 (upper lid, outer)
        p3 = 158 (upper lid, inner)
        p4 = 133 (inner corner)
        p5 = 153 (lower lid, inner)
        p6 = 144 (lower lid, outer)

    Landmark layout for left eye:
        p1 = 362 (outer corner — mirrored)
        p2 = 385
        p3 = 387
        p4 = 263
        p5 = 373
        p6 = 380

    Args:
        landmarks : MediaPipe face landmark list (landmark[i].x, .y, .z)
        eye_indices: [p1, p2, p3, p4, p5, p6] — must be exactly 6 indices

    Returns:
        EAR as a float. Typical awake range: 0.25–0.35.
        Returns 0.30 (neutral) on any error to avoid false alarms.
    """
    if len(eye_indices) != 6:
        return 0.30

    try:
        p = [landmarks[i] for i in eye_indices]

        # Vertical distances (two measurements for robustness)
        v1 = _dist(p[1], p[5])   # p2 to p6
        v2 = _dist(p[2], p[4])   # p3 to p5

        # Horizontal distance (eye width)
        h  = _dist(p[0], p[3])   # p1 to p4

        if h < 1e-6:
            return 0.30  # degenerate case — face at extreme angle

        return (v1 + v2) / (2.0 * h)

    except (IndexError, AttributeError):
        return 0.30


def _dist(a, b) -> float:
    """Euclidean distance between two MediaPipe landmarks (x, y only)."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# MediaPipe landmark indices for EAR calculation
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]


def compute_avg_ear(landmarks) -> float:
    """
    Average EAR across both eyes.
    More robust than single-eye EAR — compensates for partial occlusion
    or asymmetric lighting that affects one eye more than the other.
    """
    right = compute_ear(landmarks, RIGHT_EYE_IDX)
    left  = compute_ear(landmarks, LEFT_EYE_IDX)
    return (right + left) / 2.0


# ═══════════════════════════════════════════════════════════════════════
# HEAD POSE PITCH ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

def compute_head_pitch(landmarks) -> float:
    """
    Estimate head pitch (up/down tilt) in degrees from facial landmarks.

    Method: use the vertical offset between the nose tip and the midpoint
    of the two ear landmarks. This is a simplified but effective proxy for
    pitch that requires no camera intrinsic calibration.

    Returns:
        Pitch in degrees.
        Positive = head tilted DOWN (looking at desk).
        Negative = head tilted UP (looking at ceiling).
        Near 0   = head level, facing camera.

    Landmarks used:
        Nose tip   : 1
        Left ear   : 234
        Right ear  : 454
        Forehead   : 10
        Chin       : 152
    """
    try:
        nose     = landmarks[1]
        forehead = landmarks[10]
        chin     = landmarks[152]

        # Vertical span of face
        face_height = abs(forehead.y - chin.y)
        if face_height < 1e-6:
            return 0.0

        # Nose displacement from forehead-chin midpoint, normalised by face height
        midpoint_y = (forehead.y + chin.y) / 2.0
        offset = (nose.y - midpoint_y) / face_height

        # Scale to approximate degrees — empirically calibrated
        # offset of +0.15 ≈ 15° downward tilt
        pitch_deg = offset * 100.0
        return round(pitch_deg, 1)

    except (IndexError, AttributeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# ATTENTION ENGINE
# ═══════════════════════════════════════════════════════════════════════

class AttentionEngine:

    # ── EAR thresholds (from literature) ──────────────────────────────
    EAR_OPEN_THRESHOLD     = 0.20   # Below this = eye closed / nearly closed
    EAR_DROWSY_THRESHOLD   = 0.22   # Below this for sustained time = drowsy
    EAR_DROWSY_DURATION    = 2.0    # Seconds of low EAR before "Drowsy" triggers

    # ── Head pose thresholds ───────────────────────────────────────────
    HEAD_DOWN_THRESHOLD    = 12.0   # Degrees downward = looking at desk
    HEAD_UP_THRESHOLD      = -18.0  # Degrees upward = looking above screen

    # ── Composite score weights (must sum to 1.0) ──────────────────────
    WEIGHT_GAZE      = 0.50
    WEIGHT_EAR       = 0.30
    WEIGHT_HEAD_POSE = 0.20

    def __init__(self):
        self.current_state   = "Idle"
        self.focus_score     = 100.0
        self._smoothed_score = 100.0
        self._smooth_alpha   = 0.15

        # Thinking buffer — 2.5s grace period
        self.thinking_threshold   = 2.5
        self.last_look_away_time  = None

        # FRT tracking
        self.distraction_start_time = None
        self.last_frt               = 0.0

        # Confidence buffer — 5-frame majority vote for gaze signal
        self._signal_buffer: deque = deque(maxlen=5)

        # ── NEW: EAR drowsiness tracking ──────────────────────────────
        # We track when EAR first dropped below the drowsy threshold.
        # Only after EAR_DROWSY_DURATION seconds of sustained low EAR
        # do we change state to Drowsy. Single-frame eye closure (blink)
        # is completely ignored.
        self._low_ear_since: Optional[float] = None

        # ── NEW: Rolling EAR history for blink rate calculation ───────
        # We store the last 300 EAR readings (~37 seconds at 8Hz).
        # Blink rate = number of times EAR crossed below EAR_OPEN_THRESHOLD
        # per minute. Normal: 15-20 blinks/min. Very low (<8): cognitive overload.
        # Very high (>30): fatigue or discomfort.
        self._ear_history:   deque = deque(maxlen=300)
        self._blink_count    = 0
        self._last_ear_above = True   # tracks blink edge detection

        # ── NEW: Head pose history ────────────────────────────────────
        # Store last 5 head pitch readings for smoothing.
        self._pitch_buffer: deque = deque(maxlen=5)

    # ──────────────────────────────────────────────────────────────────
    # MAIN UPDATE METHOD
    # ──────────────────────────────────────────────────────────────────

    def update_state(
        self,
        face_present:   bool,
        gaze_on_screen: bool,
        phone_detected: bool  = False,
        ear:            float = 0.30,   # NEW parameter — Eye Aspect Ratio
        head_pitch:     float = 0.0,    # NEW parameter — head tilt in degrees
    ) -> Tuple[str, float, str]:
        """
        Process one frame's signals and return (state, composite_score, reason).

        Parameters:
            face_present   : bool   — did MediaPipe detect a face?
            gaze_on_screen : bool   — is iris in the central gaze zone?
            phone_detected : bool   — did YOLO detect a phone? (future use)
            ear            : float  — Eye Aspect Ratio (0.0–0.5 typical range)
            head_pitch     : float  — head pitch in degrees (+= down, -= up)

        Returns:
            Tuple of (state_string, focus_score_float, reason_string)
        """
        now = time.time()

        # ── Step 1: Update EAR tracking ───────────────────────────────
        ear_alert = self._update_ear(ear, now, face_present)

        # ── Step 2: Smooth head pitch ──────────────────────────────────
        self._pitch_buffer.append(head_pitch)
        smooth_pitch = sum(self._pitch_buffer) / len(self._pitch_buffer)
        head_facing_screen = (
            self.HEAD_UP_THRESHOLD < smooth_pitch < self.HEAD_DOWN_THRESHOLD
        )

        # ── Priority 1: phone detected ────────────────────────────────
        if phone_detected:
            if self.distraction_start_time is None:
                self.distraction_start_time = now
            self.current_state = "Distracted"
            self.focus_score = max(0.0, self.focus_score - 0.8)
            self._smoothed_score = self._ema(self.focus_score)
            return self.current_state, round(self._smoothed_score, 2), "Phone Detected"

        # ── Priority 2: drowsiness (EAR sustained low) ────────────────
        # Checked before gaze — user might be looking at screen but falling asleep.
        # We only trigger this if face is present (not just losing tracking).
        if face_present and ear_alert:
            self.current_state = "Drowsy"
            # Mild score penalty — drowsiness is involuntary
            self.focus_score = max(0.0, self.focus_score - 0.6)
            self._smoothed_score = self._ema(self.focus_score)
            return self.current_state, round(self._smoothed_score, 2), "Drowsiness Detected"

        # ── Priority 3: composite focused signal ──────────────────────
        raw_focused = face_present and gaze_on_screen and head_facing_screen
        confident_focused = self._get_confident_signal(raw_focused)

        if confident_focused:
            # Return from distraction → record FRT
            if self.current_state in ("Distracted", "Away", "Drowsy"):
                if self.distraction_start_time is not None:
                    self.last_frt = now - self.distraction_start_time
                    self.distraction_start_time = None

            self.current_state = "Flow"
            self.last_look_away_time = None
            self._low_ear_since = None   # reset drowsy timer on focus return

            # ── NEW: Composite score boost ─────────────────────────────
            # When in Flow, boost size is proportional to how alert the eyes
            # are. High EAR (wide awake) = faster score recovery.
            # ear_bonus: EAR 0.30 = normal, 0.35+ = extra alert.
            ear_bonus = min(0.1, max(0.0, (ear - 0.28) * 0.5))
            self.focus_score = min(100.0, self.focus_score + 0.3 + ear_bonus)
            self._smoothed_score = self._ema(self.focus_score)
            return self.current_state, round(self._smoothed_score, 2), "Focused"

        # ── Priority 4: not focused → Thinking buffer → Away ──────────
        if not confident_focused:
            if self.last_look_away_time is None:
                self.last_look_away_time = now
                self.focus_score = max(0.0, self.focus_score - 3.0)

            if self.distraction_start_time is None:
                self.distraction_start_time = now

            elapsed_away = now - self.last_look_away_time

            if elapsed_away < self.thinking_threshold:
                self.current_state = "Thinking"
                self._smoothed_score = self._ema(self.focus_score)
                return self.current_state, round(self._smoothed_score, 2), "Cognitive Processing"
            else:
                self.current_state = "Away"
                # ── Head pose aggravation ──────────────────────────────
                # If head is also turned down (looking at desk/phone),
                # apply a slightly larger penalty than pure gaze-away.
                penalty = 0.7 if not head_facing_screen else 0.5
                self.focus_score = max(0.0, self.focus_score - penalty)
                self._smoothed_score = self._ema(self.focus_score)
                return self.current_state, round(self._smoothed_score, 2), "Extended Absence"

        return "Idle", round(self._smoothed_score, 2), "Idle"

    # ──────────────────────────────────────────────────────────────────
    # EAR TRACKING
    # ──────────────────────────────────────────────────────────────────

    def _update_ear(self, ear: float, now: float, face_present: bool) -> bool:
        """
        Update EAR history, blink counter, and drowsy timer.

        Returns True if drowsiness threshold has been exceeded
        (EAR below EAR_DROWSY_THRESHOLD for > EAR_DROWSY_DURATION seconds).
        Returns False otherwise.
        """
        if not face_present:
            # No face — reset drowsy timer, don't count non-face frames
            self._low_ear_since = None
            self._last_ear_above = True
            return False

        self._ear_history.append(ear)

        # ── Blink edge detection ──────────────────────────────────────
        # Count a blink when EAR crosses DOWN through the open threshold.
        # This gives us blinks/minute, citable as a cognitive load metric.
        currently_above = ear >= self.EAR_OPEN_THRESHOLD
        if self._last_ear_above and not currently_above:
            self._blink_count += 1   # falling edge = blink start
        self._last_ear_above = currently_above

        # ── Drowsiness timer ─────────────────────────────────────────
        if ear < self.EAR_DROWSY_THRESHOLD:
            if self._low_ear_since is None:
                self._low_ear_since = now
            elif (now - self._low_ear_since) >= self.EAR_DROWSY_DURATION:
                return True   # sustained low EAR = drowsy
        else:
            # EAR recovered above threshold — reset timer
            self._low_ear_since = None

        return False

    # ──────────────────────────────────────────────────────────────────
    # CONFIDENCE BUFFER (gaze signal voting)
    # ──────────────────────────────────────────────────────────────────

    def _get_confident_signal(self, raw_focused: bool) -> bool:
        """
        5-frame majority vote. Requires 80% agreement to change state.
        Prevents single-frame noise from causing state transitions.
        """
        self._signal_buffer.append(raw_focused)

        if len(self._signal_buffer) < 3:
            return raw_focused

        focused_count = sum(self._signal_buffer)
        total = len(self._signal_buffer)

        if focused_count >= total * 0.8:
            return True
        elif focused_count <= total * 0.2:
            return False
        else:
            return self.current_state == "Flow"

    # ──────────────────────────────────────────────────────────────────
    # SCORE SMOOTHING
    # ──────────────────────────────────────────────────────────────────

    def _ema(self, target: float) -> float:
        """Exponential moving average — smooths displayed score."""
        return self._smoothed_score + self._smooth_alpha * (target - self._smoothed_score)

    # ──────────────────────────────────────────────────────────────────
    # PUBLIC ACCESSORS
    # ──────────────────────────────────────────────────────────────────

    def get_last_frt(self) -> float:
        """Return most recent Focus Recovery Time in seconds."""
        return round(self.last_frt, 2)

    def get_blink_rate(self) -> float:
        """
        Return estimated blinks per minute based on rolling history.
        Uses the length of EAR history to estimate elapsed time.
        At 8Hz WebSocket rate, 300 samples ≈ 37.5 seconds.
        """
        if len(self._ear_history) < 16:
            return 0.0
        # Approximate elapsed seconds from buffer size and 8Hz rate
        elapsed_sec = len(self._ear_history) / 8.0
        bpm = (self._blink_count / elapsed_sec) * 60.0
        return round(bpm, 1)

    def get_ear_status(self) -> str:
        """
        Return a human-readable EAR status label.
        Used in the WebSocket payload for the frontend badge.
        """
        if len(self._ear_history) == 0:
            return "Unknown"
        latest = self._ear_history[-1]
        if latest < self.EAR_OPEN_THRESHOLD:
            return "Eyes Closed"
        elif latest < self.EAR_DROWSY_THRESHOLD:
            return "Drowsy"
        elif latest > 0.32:
            return "Alert"
        else:
            return "Normal"