"""API cost tracking for Gemini and Sarvam."""

from dataclasses import dataclass
from typing import ClassVar

# USD to INR rate (July 2026)
USD_TO_INR = 95.4

# Pricing verified July 2026 against official pages.
# Gemini: https://ai.google.dev/gemini-api/docs/pricing
# Sarvam: https://docs.sarvam.ai/api-reference-docs/pricing
PRICING = {
    "gemini-2.5-flash": {
        "input": 0.30,   # $ per 1M input tokens
        "output": 2.50,  # $ per 1M output tokens
    },
    "gemini-2.5-flash-preview-tts": {
        # Priced per token, not per character.
        "input": 0.50,    # $ per 1M input text tokens
        "output": 10.00,  # $ per 1M audio output tokens (~25 tokens/sec of audio)
    },
    # Sarvam AI bulbul:v2 (Rs 15 per 10,000 chars)
    "sarvam-bulbul-v2": {
        "per_char_inr": 0.0015,  # Rs 15 / 10,000 chars
        "per_char_usd": 0.0015 / USD_TO_INR,
    },
}


@dataclass
class CostTracker:
    """Track API costs across a session (Gemini + Sarvam)."""

    translation_input_tokens: int = 0
    translation_output_tokens: int = 0
    tts_gemini_characters: int = 0
    tts_gemini_input_tokens: int = 0
    tts_gemini_audio_tokens: int = 0
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

    def add_tts(
        self,
        characters: int,
        provider: str = "gemini",
        input_tokens: int = 0,
        audio_tokens: int = 0,
    ):
        """Record a TTS API call."""
        if provider == "sarvam":
            self.tts_sarvam_characters += characters
            self.tts_sarvam_calls += 1
        else:
            self.tts_gemini_characters += characters
            self.tts_gemini_input_tokens += input_tokens
            self.tts_gemini_audio_tokens += audio_tokens
            self.tts_gemini_calls += 1

    @property
    def translation_cost(self) -> float:
        """Calculate translation cost in USD."""
        pricing = PRICING["gemini-2.5-flash"]
        input_cost = (self.translation_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.translation_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @property
    def tts_gemini_cost(self) -> float:
        """Calculate Gemini TTS cost in USD from actual token usage."""
        pricing = PRICING["gemini-2.5-flash-preview-tts"]
        input_cost = (self.tts_gemini_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.tts_gemini_audio_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

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

    @property
    def total_cost_inr(self) -> float:
        """Total cost in INR."""
        return self.total_cost * USD_TO_INR

    def summary(self) -> dict:
        """Get cost summary."""
        result = {
            "translation": {
                "input_tokens": self.translation_input_tokens,
                "output_tokens": self.translation_output_tokens,
                "cost_usd": round(self.translation_cost, 6),
            },
            "total_cost_usd": round(self.total_cost, 6),
            "total_cost_inr": round(self.total_cost_inr, 2),
        }
        if self.tts_gemini_calls > 0:
            result["tts_gemini"] = {
                "characters": self.tts_gemini_characters,
                "input_tokens": self.tts_gemini_input_tokens,
                "audio_tokens": self.tts_gemini_audio_tokens,
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
            lines.append(
                f"TTS (Gemini): {self.tts_gemini_audio_tokens:,} audio tokens "
                f"({self.tts_gemini_calls} calls) = ${self.tts_gemini_cost:.4f}"
            )
        if self.tts_sarvam_calls > 0:
            lines.append(f"TTS (Sarvam): {self.tts_sarvam_characters:,} chars ({self.tts_sarvam_calls} calls) = Rs {self.tts_sarvam_cost_inr:.2f} (${self.tts_sarvam_cost:.4f})")
        lines.append(f"Total: ${self.total_cost:.4f} (Rs {self.total_cost_inr:.2f})")
        return "\n".join(lines)
