"""
chatbot.py — CogniFlow Local LLM Interface  (v3 — CogniFlow Logic v2 Integrated)
Calls Ollama running llama3.2:1b on localhost.

WHAT'S NEW (v3):
    - Nudge prompts updated to reflect v2 state architecture:
        · Head pose is now the PRIMARY gate (not gaze)
        · THINKING is now strictly 5°–18° yaw / 5°–12° pitch deviation
        · DISTRACTED distinguishes between "head_turn" (hard gate, > 18° yaw)
          and legacy soft signals (phone, multiple_persons)
        · DROWSY is EAR-only (< 0.18 for 16 frames) — no longer blended into FocusScore
        · ABSENT = no face detected
    - get_nudge() accepts an optional `context` dict for richer, signal-aware prompts
    - answer_query() unchanged (bidirectional querying spec)
    - _call() unchanged (shared Ollama wrapper with timeout + graceful fallback)

STATE REFERENCE (v2 gate order):
    Gate 1  ABSENT          — no face detected (YOLO)
    Gate 2  DISTRACTED      — yaw > 18° OR pitch > 12° (PnP head pose, instant, no buffer)
                            — also: phone / multiple persons (YOLO, soft gate)
    Gate 3  DROWSY          — EAR < 0.18 for 16 consecutive frames
    Gate 4  THINKING        — 5°–18° yaw OR 5°–12° pitch, sustained > 2.5s grace
    Gate 5  FLOW            — all gates clear, FocusScore climbing
    Gate 5b AWAY            — face visible but no engagement signals (soft fallback)

TRANSITION BUFFER (asymmetric, frame-count based @ 8Hz):
    Leave FLOW  → 2 frames (~250ms)  [fast — distraction onset is rapid]
    Enter FLOW  → 3 frames (~375ms)  [slow — requires sustained engagement]
    THINKING grace period: 2.5s before escalating to DISTRACTED
    DROWSY: 16 consecutive frames below EAR 0.18

FocusScore:
    Floor = 0  |  Ceiling = 100  |  Entry penalty = −3.0 (on state boundary)
    EAR contributes a confidence ×0.5 penalty in the 0.18–0.22 range only;
    it has NO direct arithmetic contribution to the FocusScore formula.
"""

import httpx


# ---------------------------------------------------------------------------
# State-to-prompt templates (v2 architecture)
# ---------------------------------------------------------------------------

_NUDGE_TEMPLATES = {
    "Flow": (
        "The user's focus state is FLOW — they are genuinely working. "
        "Reason: {reason}. "
        "Write ONE short, lightly sarcastic sentence acknowledging this rare achievement "
        "to keep the momentum going. Address the user as Aditya if natural."
    ),
    "Thinking": (
        "The user's focus state is THINKING — head deviated 5°–18° from centre "
        "(small angular offset, likely reading a second monitor or taking notes). "
        "Reason: {reason}. "
        "Write ONE witty sentence nudging them to stay on task without being harsh. "
        "Address the user as Aditya if natural."
    ),
    "Distracted_head_turn": (
        "The user's focus state is DISTRACTED — a large head turn (> 18° yaw) "
        "was detected by the head-pose tracker. They have physically turned away. "
        "Reason: {reason}. "
        "Write ONE sharp, sarcastic sentence calling them back to the screen. "
        "Address the user as Aditya if natural."
    ),
    "Distracted_phone": (
        "The user's focus state is DISTRACTED — a phone was detected in frame. "
        "Reason: {reason}. "
        "Write ONE sarcastic sentence specifically about phone distraction. "
        "Address the user as Aditya if natural."
    ),
    "Distracted_multiple_persons": (
        "The user's focus state is DISTRACTED — multiple people are in frame. "
        "Reason: {reason}. "
        "Write ONE dry, sarcastic sentence about the uninvited audience. "
        "Address the user as Aditya if natural."
    ),
    "Drowsy": (
        "The user's focus state is DROWSY — their Eye Aspect Ratio has been below 0.18 "
        "for 16 consecutive frames (~2 seconds at 8Hz). "
        "Reason: {reason}. "
        "Write ONE sarcastic sentence about their eyelids staging a protest. "
        "Address the user as Aditya if natural."
    ),
    "Away": (
        "The user's focus state is AWAY — face is visible but no engagement signals detected. "
        "Reason: {reason}. "
        "Write ONE dry sentence about the desk missing them. "
        "Address the user as Aditya if natural."
    ),
    "Absent": (
        "The user's focus state is ABSENT — no face detected in frame at all. "
        "Reason: {reason}. "
        "Write ONE short, sardonic sentence about the empty chair. "
        "Address the user as Aditya if natural."
    ),
}

_NUDGE_FALLBACKS = {
    "Flow":                       "Look at you actually working — don't jinx it.",
    "Thinking":                   "Thinking hard or hardly thinking, Aditya?",
    "Distracted_head_turn":       "The task is on the screen you're NOT looking at.",
    "Distracted_phone":           "Put the phone down, Aditya.",
    "Distracted_multiple_persons":"Who invited the extra person?",
    "Drowsy":                     "Your eyelids are filing a complaint against you.",
    "Away":                       "Your desk misses you. Deeply.",
    "Absent":                     "Back to work, Aditya. The project won't finish itself.",
}


def _resolve_distracted_substate(reason: str) -> str:
    """
    Map a DISTRACTED reason string to the correct sub-state key.

    The v2 architecture distinguishes three DISTRACTED causes:
      - head_turn  : PnP head pose gate (yaw > 18° or pitch > 12°) — hard, instant
      - phone      : YOLO phone detection — soft gate
      - multiple   : YOLO multi-person detection — soft gate
    """
    reason_lower = reason.lower()
    if "phone" in reason_lower:
        return "Distracted_phone"
    if "person" in reason_lower or "multiple" in reason_lower:
        return "Distracted_multiple_persons"
    # Default: head-turn (the primary v2 distraction gate)
    return "Distracted_head_turn"


class CogniBot:
    def __init__(self):
        self.url   = "http://localhost:11434/api/generate"
        self.model = "llama3.2:1b"

    async def get_nudge(self, state: str, reason: str) -> str:
        """
        Returns a 1-sentence sarcastic motivational nudge based on the v2 state.

        Args:
            state:  One of: Flow, Thinking, Distracted, Drowsy, Away, Absent
            reason: Human-readable reason string from the detection pipeline
                    (e.g. "yaw=23.4°", "phone detected", "EAR=0.15 for 16 frames")

        State → sub-state resolution (v2):
            Distracted + "phone"    → Distracted_phone
            Distracted + "person"   → Distracted_multiple_persons
            Distracted (default)    → Distracted_head_turn  ← primary v2 gate

        Examples by state:
            Flow                   → "Look at you actually working — don't jinx it."
            Thinking               → "Thinking hard or hardly thinking?"
            Distracted (head turn) → "The task is on the screen you're NOT looking at."
            Distracted (phone)     → "Put the phone down, Aditya."
            Distracted (persons)   → "Who invited the extra person?"
            Drowsy                 → "Your eyelids are filing a complaint against you."
            Away                   → "Your desk misses you. Deeply."
            Absent                 → "Back to work, Aditya. The project won't finish itself."
        """
        # Resolve DISTRACTED into its v2 sub-state
        if state == "Distracted":
            state_key = _resolve_distracted_substate(reason)
        else:
            state_key = state

        template = _NUDGE_TEMPLATES.get(state_key)
        fallback  = _NUDGE_FALLBACKS.get(state_key,
                        "Back to work, Aditya. The project won't finish itself.")

        if template is None:
            return fallback

        prompt = template.format(reason=reason) + (
            " No preamble. No explanation. Just the sentence."
        )
        return await self._call(prompt, timeout=15.0, fallback=fallback)

    async def answer_query(self, user_text: str) -> str:
        """
        SPEC: Bidirectional Querying — answer a freeform question from the user.
        Called when the frontend sends {"command": "USER_QUERY", "text": "..."}.

        The LLM acts as a study assistant — concise, helpful, on-topic.
        Response capped at 2-3 sentences to keep latency acceptable on i5 CPU.
        """
        prompt = (
            "You are CogniBot, a concise study assistant built into a focus-tracking app. "
            "Answer the following question in 2-3 sentences maximum. "
            "Be direct and helpful. No preamble.\n\n"
            f"Question: {user_text}"
        )
        return await self._call(prompt, timeout=20.0,
                                fallback="I couldn't process that right now. Please try again.")

    async def _call(self, prompt: str, timeout: float, fallback: str) -> str:
        """Shared Ollama call with timeout and graceful fallback."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json={
                        "model":  self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 80,   # cap tokens for speed on i5
                            "temperature": 0.7,
                        },
                    },
                    timeout=timeout,
                )
                return response.json().get("response", fallback).strip()
        except Exception:
            return fallback