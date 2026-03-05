"""
main.py — CogniFlow backend  (v3 — EAR + Head Pose)
Python 3.9+ compatible

═══════════════════════════════════════════════════════════════════════
WHAT'S NEW IN THIS VERSION
═══════════════════════════════════════════════════════════════════════

AIWorker now extracts THREE signals per frame instead of two:
  1. Gaze (iris X/Y position)          ← was already present
  2. EAR  (Eye Aspect Ratio)           ← NEW — drowsiness / fatigue
  3. Head pitch (up/down tilt)         ← NEW — desk/phone looking

CSVLogger now logs FIVE columns instead of four:
  timestamp | focusScore | currentState | recoveryLatency
  | earValue | headPitch           ← NEW — for quantitative evaluation

WebSocket payload now includes SEVEN fields instead of six:
  focusScore | currentState | recoveryLatency | eyePosition
  | faceDetected | nudge
  | earValue | blinkRate | headPitch | earStatus   ← NEW

These additions directly address the paper's requirement for
measurable, citable metrics (EAR is the most-cited metric in the
drowsiness and attention literature).
═══════════════════════════════════════════════════════════════════════
"""

import csv
import cv2
import asyncio
import json
import queue
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import mediapipe as mp
import numpy as np

from engine import AttentionEngine, compute_avg_ear, compute_head_pitch
from report_generator import ReportGenerator
from chatbot import CogniBot

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# CAMERA BUFFER
# ═══════════════════════════════════════════════════════════════════
class CameraBuffer:
    def __init__(self, width: int = 320, height: int = 240):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self._frame: Optional[np.ndarray] = None
        self._lock  = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="CameraCapture").start()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            time.sleep(0.033)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        time.sleep(0.1)
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════
# AI WORKER  (v3 — EAR + Head Pose added)
# ═══════════════════════════════════════════════════════════════════
class AIWorker:
    def __init__(self):
        self._in_q: queue.Queue = queue.Queue(maxsize=2)
        self._result: Dict[str, Any] = {
            "face_present": False,
            "gaze_on":      False,
            "iris_x":       0.5,
            "iris_y":       0.5,
            # ── NEW fields ────────────────────────────────────────────
            "ear":          0.30,   # Eye Aspect Ratio (0.0–0.5 range)
            "head_pitch":   0.0,    # Head tilt in degrees
        }
        self._lock    = threading.Lock()
        self._running = True
        self._frame_count = 0

        threading.Thread(target=self._loop, daemon=True, name="AIWorker").start()

    def _loop(self):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,       # Required for iris landmarks (468+)
            max_num_faces=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )

        while self._running:
            try:
                frame: np.ndarray = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Frame skip — process every 2nd frame to save CPU
            self._frame_count += 1
            if self._frame_count % 2 != 0:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            face_present = bool(results.multi_face_landmarks)
            gaze_on      = False
            iris_x       = 0.5
            iris_y       = 0.5
            ear          = 0.30   # default to neutral (won't trigger drowsy)
            head_pitch   = 0.0

            if face_present:
                lm = results.multi_face_landmarks[0].landmark

                # ── Iris / Gaze ────────────────────────────────────────
                # Landmark 468 = right iris centre (requires refine_landmarks=True)
                iris   = lm[468]
                iris_x = iris.x
                iris_y = iris.y

                # Gaze zone: central 30% horizontally, central 50% vertically
                # Tuned for a straight-ahead laptop webcam
                gaze_on = (0.35 < iris_x < 0.65) and (0.25 < iris_y < 0.75)

                # ── NEW: EAR calculation ───────────────────────────────
                # compute_avg_ear() is imported from engine.py.
                # It averages left and right eye EAR for robustness.
                # Typical awake range: 0.25–0.35.
                # Below 0.20 = eye closed. Below 0.22 for 2s = drowsy.
                ear = compute_avg_ear(lm)

                # ── NEW: Head Pose (Pitch) ─────────────────────────────
                # compute_head_pitch() is imported from engine.py.
                # Returns degrees: positive = looking down, negative = up.
                # Thresholds in engine.py: down > 12°, up < -18°.
                head_pitch = compute_head_pitch(lm)

            with self._lock:
                self._result = {
                    "face_present": face_present,
                    "gaze_on":      gaze_on,
                    "iris_x":       iris_x,
                    "iris_y":       iris_y,
                    "ear":          round(ear, 4),
                    "head_pitch":   round(head_pitch, 2),
                }

        face_mesh.close()

    def submit(self, frame: np.ndarray):
        try:
            self._in_q.put_nowait(frame)
        except queue.Full:
            pass

    def get_result(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._result)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════
# CSV LOGGER  (v3 — EAR and head pitch columns added)
# ═══════════════════════════════════════════════════════════════════
class CSVLogger:
    """
    Writes one row per WebSocket tick to Session_Data.csv.

    NEW COLUMNS vs v2:
        earValue   — raw EAR reading for this tick (for quantitative analysis)
        headPitch  — head pitch in degrees (for quantitative analysis)

    These columns are what make quantitative evaluation possible.
    You can load this CSV and compute:
        - Mean EAR over session (alertness baseline)
        - Time-series of EAR (drowsiness onset detection)
        - Correlation between head pitch and focus score
        - Blink rate per minute (blink count from EAR history)
    """
    COLUMNS = [
        "timestamp",
        "focusScore",
        "currentState",
        "recoveryLatency",
        "earValue",       # NEW
        "headPitch",      # NEW
    ]

    def __init__(self, path: str = "Session_Data.csv"):
        self.path = path
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(
        self,
        focus_score: float,
        state:       str,
        frt:         float,
        ear:         float = 0.30,   # NEW
        head_pitch:  float = 0.0,    # NEW
    ):
        row = {
            "timestamp":       datetime.now().strftime("%H:%M:%S"),
            "focusScore":      round(focus_score, 2),
            "currentState":    state,
            "recoveryLatency": round(frt, 2),
            "earValue":        round(ear, 4),    # NEW
            "headPitch":       round(head_pitch, 2),  # NEW
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)


# ═══════════════════════════════════════════════════════════════════
# Singletons
# ═══════════════════════════════════════════════════════════════════
camera    = CameraBuffer()
ai_worker = AIWorker()
reporter  = ReportGenerator()
bot       = CogniBot()


# ═══════════════════════════════════════════════════════════════════
# VIDEO STREAM  GET /video_feed
# ═══════════════════════════════════════════════════════════════════
def _generate_frames():
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70]
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.04)
            continue
        ret, buf = cv2.imencode(".jpg", cv2.flip(frame, 1), encode_params)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.04)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ═══════════════════════════════════════════════════════════════════
# COGNITIVE WEBSOCKET  WS /cognitive-stream
# ═══════════════════════════════════════════════════════════════════
@app.websocket("/cognitive-stream")
async def cognitive_stream(websocket: WebSocket):
    await websocket.accept()
    session_active = True

    engine        = AttentionEngine()
    logger        = CSVLogger()
    last_state    = "Idle"
    current_nudge = "Monitoring your focus. Stay sharp."

    async def _listen():
        nonlocal session_active
        try:
            while session_active:
                raw = await websocket.receive_text()
                if json.loads(raw).get("command") == "STOP_SESSION":
                    session_active = False
        except (WebSocketDisconnect, Exception):
            session_active = False

    listener = asyncio.create_task(_listen())

    try:
        while session_active:
            frame = camera.get_frame()
            if frame is not None:
                ai_worker.submit(frame)

            ai = ai_worker.get_result()

            # ── Call engine with ALL THREE signals ────────────────────
            # Previously: engine.update_state(face_present, gaze_on_screen)
            # Now:        engine.update_state(face_present, gaze_on_screen,
            #                                 ear=..., head_pitch=...)
            state, score, reason = engine.update_state(
                face_present   = ai["face_present"],
                gaze_on_screen = ai["gaze_on"],
                phone_detected = False,
                ear            = ai["ear"],         # NEW
                head_pitch     = ai["head_pitch"],  # NEW
            )

            frt        = engine.get_last_frt()
            blink_rate = engine.get_blink_rate()   # NEW
            ear_status = engine.get_ear_status()   # NEW

            # ── Log to CSV with new columns ────────────────────────────
            logger.log(
                score, state, frt,
                ear        = ai["ear"],         # NEW
                head_pitch = ai["head_pitch"],  # NEW
            )

            # ── Nudge on state change ─────────────────────────────────
            if state != last_state:
                last_state = state
                try:
                    current_nudge = await asyncio.wait_for(
                        bot.get_nudge(state, reason), timeout=12.0
                    )
                except Exception:
                    current_nudge = "Stay focused — you've got this."

            # ── WebSocket payload (v3) ────────────────────────────────
            # NEW fields: earValue, blinkRate, headPitch, earStatus
            # These power the new EAR gauge and blink rate display in
            # the frontend sidebar (CogniFlowSidebar.tsx v3).
            await websocket.send_json({
                # ── Existing fields (unchanged) ───────────────────────
                "focusScore":      score,
                "currentState":    state,
                "recoveryLatency": frt,
                "eyePosition":     {"x": ai["iris_x"], "y": ai["iris_y"]},
                "faceDetected":    ai["face_present"],
                "gazeOnScreen":    ai["gaze_on"],
                "nudge":           current_nudge,
                # ── NEW fields ────────────────────────────────────────
                "earValue":        ai["ear"],          # 0.0–0.5, raw EAR
                "blinkRate":       blink_rate,         # blinks per minute
                "headPitch":       ai["head_pitch"],   # degrees
                "earStatus":       ear_status,         # "Alert"/"Normal"/"Drowsy"/"Eyes Closed"
            })

            await asyncio.sleep(0.125)   # 8 Hz

    except WebSocketDisconnect:
        pass
    finally:
        session_active = False
        listener.cancel()
        try:
            reporter.generate()
        except Exception as e:
            print(f"[ReportGenerator] {e}")


@app.on_event("shutdown")
async def _shutdown():
    ai_worker.stop()
    camera.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")