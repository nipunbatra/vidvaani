"""API cost tracking for Gemini and Sarvam."""

from dataclasses import dataclass, field
from typing import ClassVar

# USD to INR rate (approximate)
USD_TO_INR = 83.0

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
    # Sarvam AI pricing (Rs 15 per 10,000 chars)
    # https://www.sarvam.ai/apis/text-to-speech
    "sarvam-bulbul-v2": {
        "per_char_inr": 0.0015,  # Rs 15 / 10,000 = Rs 0.0015 per char
        "per_char_usd": 0.0015 / USD_TO_INR,  # ~$0.000018 per char
    },
}


@dataclass
class CostTracker:
    """Track API costs across a session (Gemini + Sarvam)."""

    translation_input_tokens: int = 0
    translation_output_tokens: int = 0
    tts_gemini_characters: int = 0
    tts_gemini_calls: int = 0
    tts_sarvam_characters: int = 0
    tts_sarvam_calls: int = 0

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

    def add_tts(self, characters: int, provider: str = "gemini"):
        """Record a TTS API call."""
        if provider == "sarvam":
            self.tts_sarvam_characters += characters
            self.tts_sarvam_calls += 1
        else:
            self.tts_gemini_characters += characters
            self.tts_gemini_calls += 1

    @property
    def translation_cost(self) -> float:
        """Calculate translation cost in USD."""
        pricing = PRICING["gemini-2.0-flash"]
        input_cost = (self.translation_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.translation_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @property
    def tts_gemini_cost(self) -> float:
        """Calculate Gemini TTS cost in USD."""
        pricing = PRICING["gemini-2.5-flash-preview-tts"]
        return self.tts_gemini_characters * pricing["per_char"]

    @property
    def tts_sarvam_cost(self) -> float:
        """Calculate Sarvam TTS cost in USD."""
        pricing = PRICING["sarvam-bulbul-v2"]
        return self.tts_sarvam_characters * pricing["per_char_usd"]

    @property
    def tts_sarvam_cost_inr(self) -> float:
        """Calculate Sarvam TTS cost in INR."""
        pricing = PRICING["sarvam-bulbul-v2"]
        return self.tts_sarvam_characters * pricing["per_char_inr"]

    @property
    def tts_cost(self) -> float:
        """Calculate total TTS cost in USD."""
        return self.tts_gemini_cost + self.tts_sarvam_cost

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return self.translation_cost + self.tts_cost

    def summary(self) -> dict:
        """Get cost summary."""
        result = {
            "translation": {
                "input_tokens": self.translation_input_tokens,
                "output_tokens": self.translation_output_tokens,
                "cost_usd": round(self.translation_cost, 6),
            },
            "total_cost_usd": round(self.total_cost, 6),
        }
        if self.tts_gemini_calls > 0:
            result["tts_gemini"] = {
                "characters": self.tts_gemini_characters,
                "calls": self.tts_gemini_calls,
                "cost_usd": round(self.tts_gemini_cost, 6),
            }
        if self.tts_sarvam_calls > 0:
            result["tts_sarvam"] = {
                "characters": self.tts_sarvam_characters,
                "calls": self.tts_sarvam_calls,
                "cost_usd": round(self.tts_sarvam_cost, 6),
                "cost_inr": round(self.tts_sarvam_cost_inr, 2),
            }
        return result

    def __str__(self) -> str:
        lines = [
            f"Translation: {self.translation_input_tokens:,} in / {self.translation_output_tokens:,} out tokens = ${self.translation_cost:.4f}"
        ]
        if self.tts_gemini_calls > 0:
            lines.append(f"TTS (Gemini): {self.tts_gemini_characters:,} chars ({self.tts_gemini_calls} calls) = ${self.tts_gemini_cost:.4f}")
        if self.tts_sarvam_calls > 0:
            lines.append(f"TTS (Sarvam): {self.tts_sarvam_characters:,} chars ({self.tts_sarvam_calls} calls) = Rs {self.tts_sarvam_cost_inr:.2f} (${self.tts_sarvam_cost:.4f})")
        lines.append(f"Total: ${self.total_cost:.4f}")
        return "\n".join(lines)
