"""Gemini API cost tracking."""

from dataclasses import dataclass, field
from typing import ClassVar

# Gemini pricing (as of Jan 2025) - per 1M tokens/characters
# https://ai.google.dev/pricing
PRICING = {
    "gemini-2.0-flash": {
        "input": 0.10,   # $0.10 per 1M input tokens
        "output": 0.40,  # $0.40 per 1M output tokens
    },
    "gemini-2.5-flash-preview-tts": {
        # TTS is priced per character
        "per_char": 0.000004,  # ~$4 per 1M characters
    },
    "gemini-1.5-flash": {
        "input": 0.075,
        "output": 0.30,
    },
}


@dataclass
class CostTracker:
    """Track Gemini API costs across a session."""

    translation_input_tokens: int = 0
    translation_output_tokens: int = 0
    tts_characters: int = 0
    tts_calls: int = 0

    # Class-level singleton for global tracking
    _instance: ClassVar["CostTracker | None"] = None

    @classmethod
    def get(cls) -> "CostTracker":
        """Get or create the global cost tracker."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the global tracker."""
        cls._instance = cls()

    def add_translation(self, input_tokens: int, output_tokens: int):
        """Record a translation API call."""
        self.translation_input_tokens += input_tokens
        self.translation_output_tokens += output_tokens

    def add_tts(self, characters: int):
        """Record a TTS API call."""
        self.tts_characters += characters
        self.tts_calls += 1

    @property
    def translation_cost(self) -> float:
        """Calculate translation cost in USD."""
        pricing = PRICING["gemini-2.0-flash"]
        input_cost = (self.translation_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.translation_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @property
    def tts_cost(self) -> float:
        """Calculate TTS cost in USD."""
        pricing = PRICING["gemini-2.5-flash-preview-tts"]
        return self.tts_characters * pricing["per_char"]

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return self.translation_cost + self.tts_cost

    def summary(self) -> dict:
        """Get cost summary."""
        return {
            "translation": {
                "input_tokens": self.translation_input_tokens,
                "output_tokens": self.translation_output_tokens,
                "cost_usd": round(self.translation_cost, 6),
            },
            "tts": {
                "characters": self.tts_characters,
                "calls": self.tts_calls,
                "cost_usd": round(self.tts_cost, 6),
            },
            "total_cost_usd": round(self.total_cost, 6),
        }

    def __str__(self) -> str:
        return (
            f"Translation: {self.translation_input_tokens:,} in / {self.translation_output_tokens:,} out tokens = ${self.translation_cost:.4f}\n"
            f"TTS: {self.tts_characters:,} chars ({self.tts_calls} calls) = ${self.tts_cost:.4f}\n"
            f"Total: ${self.total_cost:.4f}"
        )
