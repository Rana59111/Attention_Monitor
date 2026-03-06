"""
chatbot.py — CogniFlow Local LLM Interface  (v2 — Bidirectional Query)
Calls Ollama running llama3.2:1b on localhost.

WHAT'S NEW:
    answer_query(text) — handles freeform user questions.
    SPEC: "Bidirectional Querying — user types question, LLM responds in real-time"

    get_nudge()   = short, sarcastic motivational nudge (state-driven, 1 sentence)
    answer_query() = freeform answer to whatever the user typed (2-3 sentences max)

Both use the same Ollama endpoint with different prompts.
Timeouts are generous (15s / 20s) because the i5-6300U generates ~3 tokens/sec.
"""

import httpx


class CogniBot:
    def __init__(self):
        self.url   = "http://localhost:11434/api/generate"
        self.model = "llama3.2:1b"

    async def get_nudge(self, state: str, reason: str) -> str:
        """
        Returns a 1-sentence sarcastic motivational nudge based on state.
        Called automatically when the attention state changes.

        Examples by state:
          Flow       → "Look at you actually working — don't jinx it."
          Thinking   → "Thinking hard or hardly thinking?"
          Away       → "The screen didn't move, but apparently you did."
          Drowsy     → "Your eyelids are filing a complaint against you."
          Distracted → "Put the phone down, Aditya." (phone)
                       "Who invited the extra person?" (multiple persons)
          Absent     → "Your desk misses you. Deeply."
        """
        prompt = (
            f"The user's focus state is: {state}. Reason: {reason}. "
            "Write exactly ONE short, sarcastic, and motivating sentence "
            "to get them back on track. No preamble. No explanation. "
            "Just the sentence. Address the user as Aditya if it feels natural."
        )
        return await self._call(prompt, timeout=15.0,
                                fallback="Back to work, Aditya. The project won't finish itself.")

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