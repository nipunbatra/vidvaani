"""Video assembly using ffmpeg."""

import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AssemblyResult:
    output_path: Path
    duration: float
    segments_used: int


def get_duration(path: Path) -> float:
    """Get duration of media file."""
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


def concatenate_audio_segments(
    segments: list[tuple[Path, float, float]],
    output_path: Path,
    total_duration: float
) -> Path:
    """Concatenate audio segments with proper timing.

    Args:
        segments: List of (audio_path, start_time, end_time) tuples
        output_path: Output audio file path
        total_duration: Total duration of the final audio

    Returns:
        Path to concatenated audio file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create silent base track
        silent_path = tmpdir / "silent.wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono:d={total_duration}",
            "-acodec", "pcm_s16le",
            str(silent_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Build complex filter for overlaying segments
        inputs = ["-i", str(silent_path)]
        filter_parts = []

        for i, (audio_path, start_time, _) in enumerate(segments):
            inputs.extend(["-i", str(audio_path)])
            # Overlay each segment at its start time
            if i == 0:
                filter_parts.append(f"[0][1]adelay={int(start_time * 1000)}|{int(start_time * 1000)}[a1]")
                prev = "a1"
            else:
                delay_ms = int(start_time * 1000)
                filter_parts.append(f"[{i + 1}]adelay={delay_ms}|{delay_ms}[d{i}]")
                filter_parts.append(f"[{prev}][d{i}]amix=inputs=2:duration=longest[a{i + 1}]")
                prev = f"a{i + 1}"

        if not filter_parts:
            # No segments, just use silence
            subprocess.run(["cp", str(silent_path), str(output_path)], check=True)
            return output_path

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{prev}]",
            "-acodec", "aac",
            "-b:a", "192k",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback: simpler concatenation
            return concatenate_audio_simple(segments, output_path, total_duration, tmpdir)

    return output_path


def concatenate_audio_simple(
    segments: list[tuple[Path, float, float]],
    output_path: Path,
    total_duration: float,
    tmpdir: Path
) -> Path:
    """Simple audio concatenation with padding."""
    # Create individual padded segments
    padded_segments = []

    for i, (audio_path, start_time, end_time) in enumerate(segments):
        padded_path = tmpdir / f"padded_{i}.wav"
        segment_duration = end_time - start_time

        # Pad with silence before and trim/extend to match duration
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-af", f"adelay={int(start_time * 1000)}|{int(start_time * 1000)},apad=whole_dur={end_time}",
            "-acodec", "pcm_s16le",
            "-ar", "24000",
            str(padded_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        padded_segments.append(padded_path)

    if not padded_segments:
        # Create silent output
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono:d={total_duration}",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    # Mix all segments
    inputs = []
    for seg in padded_segments:
        inputs.extend(["-i", str(seg)])

    n = len(padded_segments)
    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", f"amix=inputs={n}:duration=longest",
        "-acodec", "aac",
        "-b:a", "192k",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    return output_path


def replace_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    keep_original_audio: bool = False,
    original_volume: float = 0.1
) -> AssemblyResult:
    """Replace video audio with new audio track.

    Args:
        video_path: Input video file
        audio_path: New audio track
        output_path: Output video file
        keep_original_audio: If True, mix original audio at low volume
        original_volume: Volume of original audio if kept (0.0-1.0)

    Returns:
        AssemblyResult with output info
    """
    video_duration = get_duration(video_path)

    if keep_original_audio:
        # Mix original and new audio
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex",
            f"[0:a]volume={original_volume}[orig];[1:a]volume=1.0[new];[orig][new]amix=inputs=2:duration=first",
            "-map", "0:v",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path)
        ]
    else:
        # Replace audio completely
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path)
        ]

    subprocess.run(cmd, check=True, capture_output=True)

    return AssemblyResult(
        output_path=output_path,
        duration=get_duration(output_path),
        segments_used=1
    )


def create_hindi_video(
    video_path: Path,
    audio_segments: list[tuple[Path, float, float]],
    output_path: Path,
    keep_original: bool = False,
    original_volume: float = 0.1
) -> AssemblyResult:
    """Create Hindi dubbed video from segments.

    Args:
        video_path: Original video file
        audio_segments: List of (audio_path, start_time, end_time)
        output_path: Output video path
        keep_original: Keep original audio at low volume
        original_volume: Volume of original audio (0.0-1.0)

    Returns:
        AssemblyResult with output info
    """
    video_duration = get_duration(video_path)

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        combined_audio_path = Path(tmp.name)

    try:
        # Concatenate all audio segments
        concatenate_audio_segments(
            audio_segments,
            combined_audio_path,
            video_duration
        )

        # Replace video audio
        result = replace_audio(
            video_path,
            combined_audio_path,
            output_path,
            keep_original_audio=keep_original,
            original_volume=original_volume
        )
        result.segments_used = len(audio_segments)

    finally:
        combined_audio_path.unlink(missing_ok=True)

    return result
