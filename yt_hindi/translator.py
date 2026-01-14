"""Translation using Gemini API."""

import os
import json
from dataclasses import dataclass
from google import genai

from .costs import CostTracker


@dataclass
class TranslatedSegment:
    start: float
    end: float
    original: str
    translated: str


def get_client() -> genai.Client:
    """Get Gemini client with API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    return genai.Client(api_key=api_key)


def translate_segments(
    segments: list[dict],
    source_lang: str = "English",
    target_lang: str = "Hindi",
    model: str = "gemini-2.0-flash"
) -> list[TranslatedSegment]:
    """Translate transcript segments to target language.

    Uses Gemini to translate while preserving timing constraints.
    The translation is optimized for spoken Hindi that matches
    the original audio duration.

    Args:
        segments: List of segments with start, end, text
        source_lang: Source language
        target_lang: Target language
        model: Gemini model to use

    Returns:
        List of TranslatedSegment with original and translated text
    """
    client = get_client()

    # Build prompt for batch translation
    segments_json = json.dumps([
        {"id": i, "start": s.start, "end": s.end, "text": s.text, "duration": s.end - s.start}
        for i, s in enumerate(segments)
    ], indent=2)

    prompt = f"""Translate the following {source_lang} transcript segments to {target_lang} (Hindi script).

CRITICAL REQUIREMENTS:
1. The Hindi translation MUST be speakable in approximately the same duration as the original
2. Use natural spoken Hindi, not formal/literary Hindi
3. Keep translations concise - Hindi speech is often slightly faster than English
4. Preserve the meaning but prioritize matching duration
5. Use common Hindi words, avoid overly Sanskritized terms
6. Numbers should be in Hindi words (एक, दो, तीन)

Input segments (with duration in seconds):
{segments_json}

Return ONLY a JSON array with the same structure, adding a "translated" field:
[{{"id": 0, "start": 0.0, "end": 2.5, "text": "original", "translated": "हिंदी अनुवाद"}}]

JSON output:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    # Track costs
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        tracker = CostTracker.get()
        tracker.add_translation(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0
        )

    # Parse response
    response_text = response.text.strip()
    # Handle markdown code blocks
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
        response_text = response_text.strip()

    translated_data = json.loads(response_text)

    return [
        TranslatedSegment(
            start=item["start"],
            end=item["end"],
            original=item["text"],
            translated=item["translated"]
        )
        for item in translated_data
    ]


def translate_text(text: str, target_lang: str = "Hindi", model: str = "gemini-2.0-flash") -> str:
    """Simple text translation for single strings."""
    client = get_client()

    response = client.models.generate_content(
        model=model,
        contents=f"Translate to {target_lang} (use Hindi script). Only output the translation, nothing else:\n\n{text}"
    )

    return response.text.strip()
