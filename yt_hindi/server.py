"""FastMCP server for YouTube to Hindi dubbing."""

import os
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP

from .pipeline import run_pipeline, PipelineResult
from .tts import generate_samples, synthesize_gemini, synthesize_edge_sync, GEMINI_VOICES
from .downloader import download_video
from .transcriber import transcribe, group_segments_by_duration
from .translator import translate_segments


mcp = FastMCP(
    "yt-hindi",
    description="YouTube to Hindi dubbing pipeline"
)


@mcp.tool()
def dub_youtube_to_hindi(
    url: str,
    output_dir: str = "./output",
    voice: str = "Kore",
    tts_backend: Literal["gemini", "edge"] = "gemini",
    keep_original_audio: bool = False,
    original_volume: float = 0.1
) -> dict:
    """Dub a YouTube video to Hindi.

    Downloads the video, transcribes with Whisper, translates with Gemini,
    generates Hindi speech, and assembles the final video.

    Args:
        url: YouTube video URL
        output_dir: Directory for output files
        voice: TTS voice (Gemini: Aoede/Charon/Fenrir/Kore/Puck, Edge: male/female)
        tts_backend: TTS backend (gemini for best quality, edge for free)
        keep_original_audio: Keep original audio at low volume in background
        original_volume: Volume of original audio if kept (0.0-1.0)

    Returns:
        Dictionary with output video path and metadata
    """
    result = run_pipeline(
        url=url,
        output_dir=Path(output_dir),
        voice=voice,
        tts_backend=tts_backend,
        keep_original_audio=keep_original_audio,
        original_volume=original_volume
    )

    return {
        "success": True,
        "output_video": str(result.output_video),
        "title": result.title,
        "duration": result.duration,
        "segments": result.segments_count,
        "transcript": str(result.transcript_path) if result.transcript_path else None,
        "translation": str(result.translation_path) if result.translation_path else None
    }


@mcp.tool()
def generate_voice_samples(output_dir: str = "./samples") -> dict:
    """Generate sample audio files for all available voices.

    Creates sample files for both Gemini and Edge TTS voices
    so you can compare and choose the best one.

    Args:
        output_dir: Directory to save sample files

    Returns:
        Dictionary with paths to generated samples
    """
    samples = generate_samples(Path(output_dir))

    return {
        "success": True,
        "samples": [
            {"backend": backend, "voice": voice, "path": str(path)}
            for backend, voice, path in samples
        ],
        "gemini_voices": GEMINI_VOICES,
        "edge_voices": ["male", "female"]
    }


@mcp.tool()
def transcribe_youtube(
    url: str,
    output_dir: str = "./output",
    model: str = "mlx-community/whisper-large-v3-turbo"
) -> dict:
    """Transcribe a YouTube video to text.

    Downloads and transcribes the video using Whisper.

    Args:
        url: YouTube video URL
        output_dir: Directory for output files
        model: Whisper model to use

    Returns:
        Dictionary with transcript and segments
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download
    download_result = download_video(url, output_path)

    # Transcribe
    transcript = transcribe(download_result.audio_path, model=model)

    # Cleanup
    download_result.audio_path.unlink(missing_ok=True)

    return {
        "success": True,
        "title": download_result.title,
        "language": transcript.language,
        "text": transcript.text,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcript.segments
        ]
    }


@mcp.tool()
def translate_to_hindi(
    text: str,
    model: str = "gemini-2.0-flash"
) -> dict:
    """Translate text to Hindi.

    Args:
        text: Text to translate
        model: Gemini model to use

    Returns:
        Dictionary with translated text
    """
    from .translator import translate_text

    translated = translate_text(text, target_lang="Hindi", model=model)

    return {
        "success": True,
        "original": text,
        "translated": translated
    }


@mcp.tool()
def text_to_speech_hindi(
    text: str,
    output_path: str,
    voice: str = "Kore",
    backend: Literal["gemini", "edge"] = "gemini"
) -> dict:
    """Convert Hindi text to speech.

    Args:
        text: Hindi text to speak
        output_path: Output audio file path
        voice: Voice to use (Gemini: Aoede/Charon/Fenrir/Kore/Puck, Edge: male/female)
        backend: TTS backend to use

    Returns:
        Dictionary with output file path
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if backend == "gemini":
        synthesize_gemini(text, output, voice)
    else:
        synthesize_edge_sync(text, output, voice)

    return {
        "success": True,
        "output": str(output),
        "voice": voice,
        "backend": backend
    }


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
