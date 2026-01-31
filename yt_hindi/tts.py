"""Text-to-Speech using Gemini API, Sarvam AI, and Edge TTS fallback."""

import asyncio
import base64
import os
import subprocess
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Callable

from google import genai
from google.genai import types
import edge_tts

from .costs import CostTracker


@dataclass
class AudioSegment:
    path: Path
    start: float
    end: float
    target_duration: float
    actual_duration: float


# Edge TTS voices (fallback)
EDGE_VOICES = {
    "male": "hi-IN-MadhurNeural",
    "female": "hi-IN-SwaraNeural",
}

# Gemini voice options (Aoede, Charon, Fenrir, Kore, Puck)
GEMINI_VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"]

# Sarvam AI voices (Bulbul v2) - native Hindi voices
SARVAM_VOICES = {
    "anushka": "anushka",   # Female
    "manisha": "manisha",   # Female
    "vidya": "vidya",       # Female
    "arya": "arya",         # Female
    "abhilash": "abhilash", # Male
    "karun": "karun",       # Male
    "hitesh": "hitesh",     # Male
}


def get_gemini_client() -> genai.Client:
    """Get Gemini client with API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    return genai.Client(api_key=api_key)


def get_audio_duration(path: Path) -> float:
    """Get duration of audio file using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def adjust_audio_duration(
    input_path: Path,
    output_path: Path,
    target_duration: float,
    min_speed: float = 0.85,
    max_speed: float = 1.25
) -> Path:
    """Adjust audio duration using ffmpeg atempo filter + padding.

    Strategy:
    - If Hindi is shorter: slow down (to min_speed) then pad with silence
    - If Hindi is longer: speed up (to max_speed) then truncate if needed
    - Preserves pitch while changing duration
    - Limits speed range for more natural sound

    Args:
        input_path: Input audio file
        output_path: Output audio file
        target_duration: Target duration in seconds
        min_speed: Minimum speed multiplier (default 0.85 = 15% slower)
        max_speed: Maximum speed multiplier (default 1.25 = 25% faster)
    """
    actual_duration = get_audio_duration(input_path)
    if actual_duration == 0:
        subprocess.run(["cp", str(input_path), str(output_path)], check=True)
        return output_path

    ratio = actual_duration / target_duration

    # If already close enough, skip adjustment
    if 0.95 <= ratio <= 1.05:
        subprocess.run(["cp", str(input_path), str(output_path)], check=True)
        return output_path

    if ratio < 1.0:
        # Hindi is shorter than target - slow down then pad
        slow_factor = max(min_speed, ratio)
        slowed_duration = actual_duration / slow_factor
        padding_needed = target_duration - slowed_duration

        if padding_needed > 0.1:
            # Slow down, normalize volume, and add padding at the end
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-filter_complex",
                f"[0:a]atempo={slow_factor},loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={target_duration}[out]",
                "-map", "[out]",
                "-t", str(target_duration),
                str(output_path)
            ]
        else:
            # Just slow down and normalize
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-filter:a", f"atempo={slow_factor},loudnorm=I=-16:TP=-1.5:LRA=11",
                "-vn", str(output_path)
            ]
    else:
        # Hindi is longer than target - speed up then truncate
        speed_factor = min(max_speed, ratio)

        # Chain atempo filters if needed (each limited to 0.5-2.0)
        atempo_filters = []
        remaining = speed_factor
        while remaining > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining /= 2.0
        atempo_filters.append(f"atempo={remaining}")

        filter_str = ",".join(atempo_filters)

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter:a", f"{filter_str},loudnorm=I=-16:TP=-1.5:LRA=11",
            "-t", str(target_duration),  # Truncate if still too long
            "-vn", str(output_path)
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def synthesize_gemini(
    text: str,
    output_path: Path,
    voice: str = "Kore",
    model: str = "gemini-2.5-flash-preview-tts",
    max_retries: int = 3
) -> Path:
    """Synthesize speech using Gemini TTS API.

    Args:
        text: Text to synthesize (Hindi)
        output_path: Output file path (will be .wav)
        voice: Voice name (Aoede, Charon, Fenrir, Kore, Puck)
        model: Gemini TTS model
        max_retries: Number of retries on failure

    Returns:
        Path to output audio file
    """
    import time

    client = get_gemini_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )

            # Extract audio data from response
            if not response.candidates or not response.candidates[0].content:
                raise ValueError("Empty response from TTS API")

            audio_data = response.candidates[0].content.parts[0].inline_data.data
            mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise

    # Track TTS cost
    tracker = CostTracker.get()
    tracker.add_tts(len(text))

    # Write raw audio data to temp file
    raw_path = output_path.with_suffix(".raw")
    raw_path.write_bytes(audio_data)

    # Gemini returns raw PCM 24kHz 16-bit mono
    # Convert to proper format using ffmpeg
    if output_path.suffix == ".mp3":
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",  # signed 16-bit little-endian
            "-ar", "24000",  # 24kHz sample rate
            "-ac", "1",  # mono
            "-i", str(raw_path),
            "-acodec", "libmp3lame",
            "-q:a", "2",
            str(output_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", "24000",
            "-ac", "1",
            "-i", str(raw_path),
            "-acodec", "pcm_s16le",
            str(output_path)
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    raw_path.unlink(missing_ok=True)

    return output_path


def synthesize_sarvam(
    text: str,
    output_path: Path,
    voice: str = "abhilash",
    pace: float = 1.0,
    pitch: float = 0.0,
    max_retries: int = 3
) -> Path:
    """Synthesize speech using Sarvam AI Bulbul v2 API.

    Args:
        text: Text to synthesize (Hindi)
        output_path: Output file path
        voice: Voice name (anushka, manisha, vidya, arya, abhilash, karun, hitesh)
        pace: Speech pace (0.5-2.0, default 1.0)
        pitch: Voice pitch (-0.75 to 0.75, default 0.0)
        max_retries: Number of retries on failure

    Returns:
        Path to output audio file
    """
    import time

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("Set SARVAM_API_KEY environment variable")

    # Normalize voice name
    voice_name = SARVAM_VOICES.get(voice.lower(), "abhilash")

    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key
    }
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": voice_name,
        "model": "bulbul:v2",
        "pace": pace,
        "pitch": pitch,
        "loudness": 1.0
    }

    for attempt in range(max_retries):
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 429:
            # Rate limited - wait and retry
            wait_time = 2 ** (attempt + 1)
            time.sleep(wait_time)
            continue
        response.raise_for_status()
        break
    else:
        response.raise_for_status()  # Raise the last error

    result = response.json()
    audio_base64 = result["audios"][0]

    # Track TTS cost (Rs 15 per 10,000 chars = Rs 0.0015/char)
    tracker = CostTracker.get()
    tracker.add_tts(len(text), provider="sarvam")

    # Decode base64 audio and save
    audio_data = base64.b64decode(audio_base64)
    wav_path = output_path.with_suffix(".wav")
    wav_path.write_bytes(audio_data)

    # Convert to mp3 if needed
    if output_path.suffix == ".mp3":
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-acodec", "libmp3lame", "-q:a", "2",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        wav_path.unlink(missing_ok=True)
    else:
        subprocess.run(["mv", str(wav_path), str(output_path)], check=True)

    return output_path


async def synthesize_edge(
    text: str,
    output_path: Path,
    voice: str = "male",
    rate: str = "+0%"
) -> Path:
    """Synthesize speech using Edge TTS (free fallback)."""
    voice_name = EDGE_VOICES.get(voice, EDGE_VOICES["male"])

    communicate = edge_tts.Communicate(text, voice_name, rate=rate)
    await communicate.save(str(output_path))
    return output_path


def synthesize_edge_sync(text: str, output_path: Path, voice: str = "male", rate: str = "+0%") -> Path:
    """Synchronous wrapper for edge TTS."""
    return asyncio.run(synthesize_edge(text, output_path, voice, rate))


def synthesize_segment(
    text: str,
    target_duration: float,
    output_path: Path,
    voice: str = "Kore",
    backend: Literal["gemini", "edge", "sarvam"] = "gemini"
) -> AudioSegment:
    """Synthesize a segment with duration matching.

    Args:
        text: Hindi text to synthesize
        target_duration: Target duration in seconds
        output_path: Output file path
        voice: Voice name (Gemini: Aoede/Charon/Fenrir/Kore/Puck, Edge: male/female, Sarvam: abhilash/anushka/etc)
        backend: TTS backend to use

    Returns:
        AudioSegment with actual duration info
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # First pass: generate at normal speed
        if backend == "gemini":
            synthesize_gemini(text, tmp_path, voice)
        elif backend == "sarvam":
            synthesize_sarvam(text, tmp_path, voice)
        else:
            synthesize_edge_sync(text, tmp_path, voice)

        actual_duration = get_audio_duration(tmp_path)

        # Adjust if needed
        if actual_duration > 0 and abs(actual_duration - target_duration) > 0.1:
            adjust_audio_duration(tmp_path, output_path, target_duration)
        else:
            subprocess.run(["cp", str(tmp_path), str(output_path)], check=True)

        final_duration = get_audio_duration(output_path)

    finally:
        tmp_path.unlink(missing_ok=True)

    return AudioSegment(
        path=output_path,
        start=0,
        end=final_duration,
        target_duration=target_duration,
        actual_duration=final_duration
    )


def generate_samples(output_dir: Path, text: str = "नमस्ते, आज का मौसम बहुत अच्छा है। यह एक परीक्षण है।"):
    """Generate sample audio files for all available voices."""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []

    # Edge TTS samples (always available)
    for voice_type in ["male", "female"]:
        path = output_dir / f"edge_{voice_type}.mp3"
        synthesize_edge_sync(text, path, voice_type)
        samples.append(("edge", voice_type, path))
        print(f"Generated: {path}")

    # Gemini TTS samples
    try:
        for voice in GEMINI_VOICES:
            path = output_dir / f"gemini_{voice.lower()}.mp3"
            synthesize_gemini(text, path, voice)
            samples.append(("gemini", voice, path))
            print(f"Generated: {path}")
    except Exception as e:
        print(f"Gemini TTS error: {e}")

    return samples


@dataclass
class TTSJob:
    """A TTS job to be processed."""
    index: int
    text: str
    start: float
    end: float
    output_path: Path


def synthesize_batch_parallel(
    jobs: list[TTSJob],
    voice: str = "Kore",
    backend: Literal["gemini", "edge", "sarvam"] = "gemini",
    max_workers: int = 5,
    progress_callback: Callable[[int, int], None] | None = None
) -> list[tuple[Path, float, float]]:
    """Synthesize multiple segments in parallel.

    Args:
        jobs: List of TTSJob with text and timing info
        voice: Voice to use
        backend: TTS backend
        max_workers: Max parallel workers (don't exceed API rate limits)
        progress_callback: Optional callback(completed, total)

    Returns:
        List of (audio_path, start, end) tuples in order
    """
    results = [None] * len(jobs)
    completed = 0

    def process_job(job: TTSJob) -> tuple[int, Path, float, float]:
        target_duration = job.end - job.start
        segment = synthesize_segment(
            job.text,
            target_duration,
            job.output_path,
            voice=voice,
            backend=backend
        )
        return job.index, segment.path, job.start, job.end

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_job, job): job for job in jobs}

        for future in as_completed(futures):
            idx, path, start, end = future.result()
            results[idx] = (path, start, end)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(jobs))

    return results
