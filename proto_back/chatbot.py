import httpx
import json

class CogniBot:
    def __init__(self):
        # Ollama's local endpoint
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3.2:1b"

    async def get_nudge(self, state, reason):
        """Asynchronously requests a nudge from the local Llama model."""
        # We use a very strict prompt to keep the generation fast on your i5.
        prompt = (f"User status: {state}. Reason: {reason}. "
                  "Write a 1-sentence, sarcastic, and motivating nudge to keep them working. "
                  "No preamble, just the quote.")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.url,
                    json={"model": self.model, "prompt": prompt, "stream": False},
                    timeout=15.0 # Increased timeout for CPU-based generation
                )
                return response.json().get("response", "Success requires focus, Aditya!")
        except Exception:
            return "Back to work! Your project won't finish itself."