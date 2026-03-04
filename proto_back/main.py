"""
main.py — CogniFlow backend
Compatible with Python 3.9+

Fixes applied
─────────────────────────────────────────────────────────────────────
BUG 1: engine.update_state called with 4 args, signature had 3  → fixed
BUG 3: returned 3-tuple unpacked as 2                           → fixed
BUG 4: Session_Data.csv never written → report always empty     → CSVLogger added
BUG 5: recoveryLatency / eyePosition never sent to frontend     → added to payload
BUG 7: CogniBot never imported or called                        → integrated
PY39:  X | None syntax replaced with Optional[X] for Python 3.9 compatibility
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

from engine import AttentionEngine
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
# CAMERA BUFFER — captures once, shares with all consumers
# ═══════════════════════════════════════════════════════════════════
class CameraBuffer:
    def __init__(self, width: int = 320, height: int = 240):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self._frame: Optional[np.ndarray] = None   # ← Python 3.9 compatible
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="CameraCapture").start()

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            time.sleep(0.033)

    def get_frame(self) -> Optional[np.ndarray]:           # ← Python 3.9 compatible
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        time.sleep(0.1)
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════
# AI WORKER — MediaPipe on its own dedicated thread
# ═══════════════════════════════════════════════════════════════════
class AIWorker:
    def __init__(self):
        self._in_q: queue.Queue = queue.Queue(maxsize=2)
        self._result: Dict[str, Any] = {
            "face_present": False,
            "gaze_on": False,
            "iris_x": 0.5,
            "iris_y": 0.5,
        }
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="AIWorker").start()

    def _loop(self):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        while self._running:
            try:
                frame: np.ndarray = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            face_present = bool(results.multi_face_landmarks)
            gaze_on, iris_x, iris_y = False, 0.5, 0.5
            if face_present:
                iris = results.multi_face_landmarks[0].landmark[468]
                iris_x, iris_y = iris.x, iris.y
                gaze_on = 0.30 < iris.x < 0.70
            with self._lock:
                self._result = {
                    "face_present": face_present,
                    "gaze_on": gaze_on,
                    "iris_x": iris_x,
                    "iris_y": iris_y,
                }
        face_mesh.close()

    def submit(self, frame: np.ndarray):
        try:
            self._in_q.put_nowait(frame)
        except queue.Full:
            pass  # Drop stale frame — better than stalling on dual-core i5

    def get_result(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._result)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════
# CSV LOGGER
# BUG 4 FIX: nothing ever wrote Session_Data.csv — report always
#            returned "Error: No session data available".
# ═══════════════════════════════════════════════════════════════════
class CSVLogger:
    COLUMNS = ["timestamp", "focusScore", "currentState", "recoveryLatency"]

    def __init__(self, path: str = "Session_Data.csv"):
        self.path = path
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, focus_score: float, state: str, frt: float):
        row = {
            "timestamp":       datetime.now().strftime("%H:%M:%S"),
            "focusScore":      round(focus_score, 2),
            "currentState":    state,
            "recoveryLatency": round(frt, 2),
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)


# ═══════════════════════════════════════════════════════════════════
# Singletons — created once at startup
# ═══════════════════════════════════════════════════════════════════
camera    = CameraBuffer()
ai_worker = AIWorker()
reporter  = ReportGenerator()
bot       = CogniBot()   # BUG 7 FIX: was never instantiated in original


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

    # Per-session objects — safe for multiple concurrent sessions
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

            # BUG 1 + BUG 3 FIX:
            # Before: state, score = engine.update_state(face, gaze, False, False)
            #         → 4 args into 3-param function + 2-unpack of 3-tuple = two crashes
            # After:  unpack all 3 values, pass phone_detected as keyword arg
            state, score, reason = engine.update_state(
                face_present=ai["face_present"],
                gaze_on_screen=ai["gaze_on"],
                phone_detected=False,
            )

            frt = engine.get_last_frt()

            # BUG 4 FIX: write one row to CSV every tick
            logger.log(score, state, frt)

            # BUG 7 FIX: call CogniBot on state change only (not every tick)
            if state != last_state:
                last_state = state
                try:
                    current_nudge = await asyncio.wait_for(
                        bot.get_nudge(state, reason), timeout=12.0
                    )
                except Exception:
                    current_nudge = "Stay focused — you've got this."

            # BUG 5 FIX: send ALL fields the frontend expects
            # Before: {"focusScore": score, "currentState": state}  ← missing 4 fields
            # After:  full payload with recoveryLatency, eyePosition, faceDetected, nudge
            await websocket.send_json({
                "focusScore":      round(score, 2),
                "currentState":    state,
                "recoveryLatency": frt,
                "eyePosition":     {"x": ai["iris_x"], "y": ai["iris_y"]},
                "faceDetected":    ai["face_present"],
                "nudge":           current_nudge,
            })

            await asyncio.sleep(0.1)   # 10 Hz data rate

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