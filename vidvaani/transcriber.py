"""Whisper transcription using mlx-whisper (Mac-optimized)."""

import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
import mlx_whisper


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    segments: list[Segment]
    language: str
    text: str


def transcribe(
    audio_path: Path,
    model: str = "mlx-community/distil-whisper-large-v3",
    progress_callback: Callable[[int, int], None] | None = None
) -> Transcript:
    """Transcribe audio file using mlx-whisper.

    Args:
        audio_path: Path to audio file (WAV recommended)
        model: Whisper model to use (mlx-community models)
        progress_callback: Optional callback(current_seconds, total_seconds)

    Returns:
        Transcript with segment timestamps
    """
    # Get audio duration first for progress tracking
    import subprocess
    duration_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip()) if result.stdout.strip() else 0

    # Track progress via segments as they come in
    segments_so_far = []
    last_end_time = 0

    def segment_callback(segment):
        nonlocal last_end_time
        segments_so_far.append(segment)
        last_end_time = segment.get("end", last_end_time)
        if progress_callback and total_duration > 0:
            progress_callback(int(last_end_time), int(total_duration))

    # mlx_whisper.transcribe with verbose=True prints progress
    # We'll capture segments as they're processed
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        word_timestamps=False,  # Faster without word timestamps
        verbose=False,
    )

    # Final progress update
    if progress_callback and total_duration > 0:
        progress_callback(int(total_duration), int(total_duration))

    # Build segment list
    sentence_segments = [
        Segment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
        for seg in result.get("segments", [])
    ]

    return Transcript(
        segments=sentence_segments,
        language=result.get("language", "en"),
        text=result.get("text", "")
    )


def group_segments_by_duration(segments: list[Segment], max_duration: float = 10.0) -> list[Segment]:
    """Group segments to not exceed max duration for better TTS quality.

    Args:
        segments: List of transcript segments
        max_duration: Maximum duration per group in seconds

    Returns:
        List of grouped segments
    """
    grouped = []
    current_text = []
    current_start = None
    current_end = None

    for seg in segments:
        if current_start is None:
            current_start = seg.start
            current_end = seg.end
            current_text = [seg.text]
        elif seg.end - current_start <= max_duration:
            current_text.append(seg.text)
            current_end = seg.end
        else:
            grouped.append(Segment(
                start=current_start,
                end=current_end,
                text=" ".join(current_text)
            ))
            current_start = seg.start
            current_end = seg.end
            current_text = [seg.text]

    if current_text:
        grouped.append(Segment(
            start=current_start,
            end=current_end,
            text=" ".join(current_text)
        ))

    return grouped
