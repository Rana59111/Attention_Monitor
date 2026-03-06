"""
main.py — CogniFlow Backend  (v5 — Full Spec Compliance)
Python 3.9+ compatible

════════════════════════════════════════════════════════════════════
WHAT'S NEW / CHANGED vs v4
════════════════════════════════════════════════════════════════════

1. YOLOWorker added
   - Runs YOLOv8-nano in a separate thread (CPU-only, ~3 FPS on i5)
   - Detects YOLO class 67 (cell phone) → sets phone_detected = True
   - Counts person detections (class 0) → sets person_count
   - Runs at 3 FPS (not 8) to avoid overwhelming the i5 CPU
   - MediaPipe and YOLO run in separate threads so neither blocks the other

2. Bidirectional Query (SPEC: "Bidirectional Querying")
   - _listen() now handles TWO message types:
       {"command": "STOP_SESSION"}   → ends session (unchanged)
       {"command": "USER_QUERY", "text": "..."}  → NEW
   - USER_QUERY is routed to bot.answer_query(text) which calls
     Llama with the user's question as a freeform prompt
   - Response is sent immediately as a separate WebSocket message:
       {"type": "query_response", "text": "..."}
   - State machine is NOT interrupted — monitoring continues

3. Alerts wired to payload (SPEC: bounce animation + drowsy alert)
   - engine.get_alerts() is called every tick
   - Full alerts dict sent in WebSocket payload under "alerts" key
   - Frontend reads alerts.low_focus  → bounce animation
   - Frontend reads alerts.drowsy     → high-priority overlay
   - Frontend reads alerts.possible_spoof → anti-spoof badge

4. person_count passed to engine
   - AIWorker now returns person_count from YOLO (default 1 if YOLO
     not yet run or face mesh found exactly 1 face)
   - engine.update_state(person_count=...) triggers Multiple Persons
     state when > 1
════════════════════════════════════════════════════════════════════
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

from engine import AttentionEngine, compute_avg_ear, compute_head_pitch_raw
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


# ════════════════════════════════════════════════════════════════════
# CAMERA BUFFER
# ════════════════════════════════════════════════════════════════════
class CameraBuffer:
    def __init__(self, width: int = 320, height: int = 240):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        self._frame:   Optional[np.ndarray] = None
        self._lock     = threading.Lock()
        self._running  = True
        threading.Thread(target=self._loop, daemon=True, name="Camera").start()

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


# ════════════════════════════════════════════════════════════════════
# MEDIAPIPE AI WORKER  (face mesh, EAR, gaze, head pose)
# ════════════════════════════════════════════════════════════════════
class AIWorker:
    """
    Runs MediaPipe FaceMesh in a background thread.
    Processes every 2nd frame to save CPU on the i5-6300U.
    Outputs: face_present, gaze_on, iris_x/y, ear, head_pitch.
    person_count from MediaPipe is always ≤ max_num_faces (3).
    """

    def __init__(self):
        self._in_q: queue.Queue = queue.Queue(maxsize=2)
        self._result: Dict[str, Any] = {
            "face_present":    False,
            "gaze_on":         False,
            "iris_x":          0.5,
            "iris_y":          0.5,
            "ear":             0.30,
            "head_pitch":      0.0,
            "mp_person_count": 0,
            # Normalised bounding box [x_min, y_min, x_max, y_max] in [0,1].
            # Computed from the convex hull of all 468 face landmarks.
            # None when no face is present.
            "face_bbox":       None,
        }
        self._lock        = threading.Lock()
        self._running     = True
        self._frame_count = 0
        threading.Thread(target=self._loop, daemon=True, name="AIWorker").start()

    def _loop(self):
        # max_num_faces=3 so we can detect when a second person enters frame
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=3,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60,
        )
        while self._running:
            try:
                frame: np.ndarray = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self._frame_count += 1
            if self._frame_count % 2 != 0:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            mp_count     = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
            face_present = mp_count >= 1
            gaze_on      = False
            iris_x       = 0.5
            iris_y       = 0.5
            ear          = 0.30
            head_pitch   = 0.0

            if face_present:
                # Always use first (primary) face for gaze/EAR/pose
                lm     = results.multi_face_landmarks[0].landmark
                iris   = lm[468]
                iris_x = iris.x
                iris_y = iris.y
                gaze_on    = (0.35 < iris_x < 0.65) and (0.25 < iris_y < 0.75)
                ear        = compute_avg_ear(lm)
                head_pitch = compute_head_pitch_raw(lm)

                # Compute tight bounding box from all 468 landmarks
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                # Add a small 3% padding so the box sits outside the face
                pad = 0.03
                face_bbox = [
                    max(0.0, min(xs) - pad),
                    max(0.0, min(ys) - pad),
                    min(1.0, max(xs) + pad),
                    min(1.0, max(ys) + pad),
                ]
            else:
                face_bbox = None

            with self._lock:
                self._result = {
                    "face_present":    face_present,
                    "gaze_on":         gaze_on,
                    "iris_x":          iris_x,
                    "iris_y":          iris_y,
                    "ear":             round(ear, 4),
                    "head_pitch":      round(head_pitch, 4),
                    "mp_person_count": mp_count,
                    "face_bbox":       face_bbox,
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


# ════════════════════════════════════════════════════════════════════
# YOLO WORKER  (phone detection + person count)
# NEW — spec requires YOLO class 67 phone detection
# ════════════════════════════════════════════════════════════════════
class YOLOWorker:
    """
    Runs YOLOv8-nano object detection in a background thread.

    Why separate thread:
        YOLO on CPU takes ~150–300ms per frame on an i5-6300U.
        Running it in the same loop as MediaPipe would drop the
        WebSocket rate from 8Hz to ~3Hz. Separate thread = both
        run at their natural rates without blocking each other.

    Detection targets:
        Class 0  = person  → person_count
        Class 67 = cell phone → phone_detected

    Frame rate:
        Processes one frame every 0.33s (~3 FPS). Sufficient for
        detecting a phone being raised to the camera — this doesn't
        need millisecond latency.

    Graceful fallback:
        If ultralytics is not installed or YOLO model download fails,
        YOLOWorker disables itself silently. The system continues
        running with phone_detected=False and person_count from
        MediaPipe only.
    """

    PHONE_CLASS  = 67   # YOLO COCO class 67 = cell phone
    PERSON_CLASS = 0    # YOLO COCO class 0  = person
    CONF_THRESH  = 0.45 # minimum confidence to count a detection

    def __init__(self):
        self._result: Dict[str, Any] = {
            "phone_detected": False,
            "yolo_person_count": 0,
        }
        self._lock    = threading.Lock()
        self._running = True
        self._enabled = False
        self._frame:  Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        threading.Thread(target=self._loop, daemon=True, name="YOLOWorker").start()

    def _loop(self):
        # Lazy import — don't crash if ultralytics not installed
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")   # nano — fastest on CPU
            model.to("cpu")
            self._enabled = True
            print("[CogniFlow] YOLOWorker: yolov8n loaded, phone detection active.")
        except Exception as e:
            print(f"[CogniFlow] YOLOWorker disabled: {e}")
            print("[CogniFlow] Install with: pip install ultralytics --break-system-packages")
            return

        while self._running:
            with self._frame_lock:
                frame = self._frame.copy() if self._frame is not None else None

            if frame is None:
                time.sleep(0.1)
                continue

            try:
                results = model(frame, verbose=False, conf=self.CONF_THRESH)
                phone_detected    = False
                yolo_person_count = 0

                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    if cls == self.PHONE_CLASS:
                        phone_detected = True
                    elif cls == self.PERSON_CLASS:
                        yolo_person_count += 1

                with self._lock:
                    self._result = {
                        "phone_detected":    phone_detected,
                        "yolo_person_count": yolo_person_count,
                    }
            except Exception as e:
                print(f"[YOLOWorker] inference error: {e}")

            time.sleep(0.33)  # ~3 FPS

    def submit(self, frame: np.ndarray):
        with self._frame_lock:
            self._frame = frame.copy()

    def get_result(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._result)

    def stop(self):
        self._running = False

    @property
    def enabled(self) -> bool:
        return self._enabled


# ════════════════════════════════════════════════════════════════════
# CSV LOGGER
# ════════════════════════════════════════════════════════════════════
class CSVLogger:
    COLUMNS = [
        "timestamp", "focusScore", "currentState",
        "recoveryLatency", "earValue", "headPitch",
        "phoneDetected", "personCount",
    ]

    def __init__(self, path: str = "Session_Data.csv"):
        self.path = path
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(self, focus_score: float, state: str, frt: float,
            ear: float = 0.30, head_pitch: float = 0.0,
            phone_detected: bool = False, person_count: int = 1):
        row = {
            "timestamp":       datetime.now().strftime("%H:%M:%S"),
            "focusScore":      round(focus_score, 2),
            "currentState":    state,
            "recoveryLatency": round(frt, 2),
            "earValue":        round(ear, 4),
            "headPitch":       round(head_pitch, 4),
            "phoneDetected":   int(phone_detected),
            "personCount":     person_count,
        }
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)


# ════════════════════════════════════════════════════════════════════
# Singletons
# ════════════════════════════════════════════════════════════════════
camera      = CameraBuffer()
ai_worker   = AIWorker()
yolo_worker = YOLOWorker()
reporter    = ReportGenerator()
bot         = CogniBot()


# ════════════════════════════════════════════════════════════════════
# OVERLAY STATE  (shared between WebSocket loop and video generator)
# ════════════════════════════════════════════════════════════════════
# The WebSocket loop knows the current attention state and face bbox.
# The video generator reads these to draw the overlay on each JPEG frame.
# Using a simple dict + lock — no queues needed, latest value wins.
_overlay_lock  = threading.Lock()
_overlay_state: Dict[str, Any] = {
    "face_bbox": None,   # [x_min, y_min, x_max, y_max] normalised, or None
    "state":     "Idle", # current attention state string
}

# Colour map: attention state → BGR colour for the bounding box
_STATE_COLOURS: Dict[str, tuple] = {
    "Flow":       (0,   220,  0),    # green   — focused
    "Thinking":   (0,   180, 255),   # amber   — grace period
    "Away":       (0,   140, 255),   # orange  — distracted gaze
    "Drowsy":     (0,    80, 255),   # red-orange — fatigue
    "Distracted": (0,    0,  255),   # red     — phone / multiple persons
    "Absent":     (128, 128, 128),   # grey    — nobody there
    "Idle":       (200, 200, 200),   # light grey — not started
}


# ════════════════════════════════════════════════════════════════════
# VIDEO STREAM
# ════════════════════════════════════════════════════════════════════
def _generate_frames():
    """
    Grabs the latest camera frame, draws a colour-coded face bounding box
    if a face is detected, mirrors the image (so it feels like a mirror),
    then JPEG-encodes and yields for the multipart HTTP stream.

    Box colour reflects current attention state:
      Green      = Flow (focused)
      Amber      = Thinking (grace period)
      Orange     = Away (gaze off screen)
      Red-orange = Drowsy
      Red        = Distracted (phone / multiple persons)
      Grey       = Absent / Idle
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 72]

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.04)
            continue

        # Mirror horizontally — feels natural for the user
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Read latest overlay state (non-blocking)
        with _overlay_lock:
            bbox  = _overlay_state["face_bbox"]
            state = _overlay_state["state"]

        if bbox is not None:
            colour    = _STATE_COLOURS.get(state, (200, 200, 200))
            thickness = 2

            # bbox is normalised [0,1] — convert to pixel coords.
            # Frame is already mirrored, so x coords need mirroring too:
            #   mirrored_x = 1.0 - original_x
            x_min_n, y_min_n, x_max_n, y_max_n = bbox
            # Mirror x
            mx_min = 1.0 - x_max_n
            mx_max = 1.0 - x_min_n

            x1 = int(mx_min * w)
            y1 = int(y_min_n * h)
            x2 = int(mx_max * w)
            y2 = int(y_max_n * h)

            # Main rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

            # Corner accent lines — same style as the reference image
            corner = 14   # length of corner tick in pixels
            t      = thickness + 1

            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner, y1), colour, t)
            cv2.line(frame, (x1, y1), (x1, y1 + corner), colour, t)
            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - corner, y1), colour, t)
            cv2.line(frame, (x2, y1), (x2, y1 + corner), colour, t)
            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + corner, y2), colour, t)
            cv2.line(frame, (x1, y2), (x1, y2 - corner), colour, t)
            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - corner, y2), colour, t)
            cv2.line(frame, (x2, y2), (x2, y2 - corner), colour, t)

            # Small state label above the box (e.g. "Flow", "Drowsy")
            label     = state.upper()
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            label_y   = max(y1 - 6, 12)
            # Dark background pill for readability
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(frame,
                          (x1, label_y - lh - 2),
                          (x1 + lw + 6, label_y + 2),
                          colour, cv2.FILLED)
            cv2.putText(frame, label,
                        (x1 + 3, label_y),
                        font, font_scale,
                        (0, 0, 0),   # black text on coloured background
                        1, cv2.LINE_AA)

        ret, buf = cv2.imencode(".jpg", frame, encode_params)
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.04)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ════════════════════════════════════════════════════════════════════
# COGNITIVE WEBSOCKET
# ════════════════════════════════════════════════════════════════════
@app.websocket("/cognitive-stream")
async def cognitive_stream(websocket: WebSocket):
    await websocket.accept()
    session_active = True

    engine        = AttentionEngine()
    logger        = CSVLogger()
    last_state    = "Idle"
    current_nudge = "CogniFlow is active. Stay sharp."

    # ── Bidirectional query listener ─────────────────────────────────
    # SPEC: "Bidirectional Querying — user types question, LLM responds"
    #
    # Runs as a concurrent asyncio task alongside the main broadcast loop.
    # Handles two message types:
    #   {"command": "STOP_SESSION"}            → ends session
    #   {"command": "USER_QUERY", "text": "…"} → query Llama, reply immediately
    #
    # The query response is sent as a SEPARATE message type so the frontend
    # can distinguish it from the regular telemetry stream:
    #   {"type": "query_response", "text": "…"}
    # ─────────────────────────────────────────────────────────────────
    async def _listen():
        nonlocal session_active
        try:
            while session_active:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                command = msg.get("command", "")

                if command == "STOP_SESSION":
                    session_active = False

                elif command == "USER_QUERY":
                    # Route to Llama as a freeform question
                    user_text = msg.get("text", "").strip()
                    if not user_text:
                        continue
                    try:
                        answer = await asyncio.wait_for(
                            bot.answer_query(user_text), timeout=20.0
                        )
                    except Exception:
                        answer = "Sorry, I couldn't process that. Try again."
                    # Send query response as its own typed message
                    await websocket.send_json({
                        "type": "query_response",
                        "text": answer,
                    })

        except (WebSocketDisconnect, Exception):
            session_active = False

    listener = asyncio.create_task(_listen())

    try:
        while session_active:
            frame = camera.get_frame()
            if frame is not None:
                ai_worker.submit(frame)
                yolo_worker.submit(frame)   # YOLO gets same frame, runs at 3FPS internally

            ai   = ai_worker.get_result()
            yolo = yolo_worker.get_result()

            # ── Merge person counts from both detectors ───────────────
            # MediaPipe gives face count (more accurate for faces).
            # YOLO gives full-body person count.
            # Use whichever reports more people (conservative approach).
            mp_count   = ai.get("mp_person_count", 1)
            yolo_count = yolo.get("yolo_person_count", 0)
            person_count = max(mp_count, yolo_count, 1)

            phone_detected = yolo.get("phone_detected", False)

            # ── Engine update ─────────────────────────────────────────
            state, score, reason = engine.update_state(
                face_present   = ai["face_present"],
                gaze_on_screen = ai["gaze_on"],
                phone_detected = phone_detected,
                ear            = ai["ear"],
                head_pitch     = ai["head_pitch"],
                person_count   = person_count,
            )

            frt          = engine.get_last_frt()
            blink_rate   = engine.get_blink_rate()
            ear_status   = engine.get_ear_status()
            calibrated   = engine.is_calibrated()
            alerts       = engine.get_alerts()

            # ── Update video overlay ──────────────────────────────────
            # Write current state + face bbox so _generate_frames() can
            # draw the colour-coded bounding box on the video stream.
            with _overlay_lock:
                _overlay_state["face_bbox"] = ai.get("face_bbox")
                _overlay_state["state"]     = state

            # ── CSV log ───────────────────────────────────────────────
            logger.log(
                score, state, frt,
                ear            = ai["ear"],
                head_pitch     = ai["head_pitch"],
                phone_detected = phone_detected,
                person_count   = person_count,
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

            # ── Telemetry broadcast ───────────────────────────────────
            # type="telemetry" lets frontend distinguish from query_response
            await websocket.send_json({
                "type":            "telemetry",       # NEW — message type tag
                # Core state
                "focusScore":      score,
                "currentState":    state,
                "recoveryLatency": frt,
                "nudge":           current_nudge,
                # Face / gaze
                "faceDetected":    ai["face_present"],
                "gazeOnScreen":    ai["gaze_on"],
                "eyePosition":     {"x": ai["iris_x"], "y": ai["iris_y"]},
                # EAR
                "earValue":        ai["ear"],
                "blinkRate":       blink_rate,
                "earStatus":       ear_status,
                # Head pose
                "headPitch":       ai["head_pitch"],
                "calibrated":      calibrated,
                # YOLO
                "phoneDetected":   phone_detected,
                "personCount":     person_count,
                "yoloActive":      yolo_worker.enabled,
                # Alerts — drive all frontend notifications
                # alerts.low_focus      → bounce animation (score < 60)
                # alerts.drowsy         → high-priority drowsy overlay
                # alerts.absent         → user left desk notification
                # alerts.possible_spoof → anti-spoof warning badge
                # alerts.multiple_persons → proctoring flag
                "alerts":          alerts,
            })

            await asyncio.sleep(0.125)  # 8 Hz

    except WebSocketDisconnect:
        pass
    finally:
        session_active = False
        listener.cancel()
        # Clear overlay so the box disappears after session ends
        with _overlay_lock:
            _overlay_state["face_bbox"] = None
            _overlay_state["state"]     = "Idle"
        try:
            reporter.generate()
        except Exception as e:
            print(f"[ReportGenerator] {e}")


@app.on_event("shutdown")
async def _shutdown():
    ai_worker.stop()
    yolo_worker.stop()
    camera.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")