"""Video assembly using ffmpeg."""

import subprocess
import tempfile
import shutil
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


def get_video_dimensions(path: Path) -> tuple[int, int]:
    """Get video width and height."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)
    except:
        return 1920, 1080  # Default


def detect_intro_offset(path: Path, min_intro_duration: float = 3.0,
                        scan_duration: float = 60.0) -> float:
    """Detect intro music/silence and return offset where speech starts.

    Uses silence detection to find the pattern:
    - Intro music (0 to X seconds)
    - Silence/pause (X to Y seconds)
    - Speech starts at Y

    Args:
        path: Path to video/audio file
        min_intro_duration: Minimum intro length to consider (ignore short silences)
        scan_duration: How many seconds to scan from start

    Returns:
        Offset in seconds where speech starts (0.0 if no intro detected)
    """
    import re

    cmd = [
        "ffmpeg", "-i", str(path),
        "-af", "silencedetect=noise=-30dB:d=0.5",
        "-t", str(scan_duration),
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    # Parse silence_end timestamps
    # Format: [silencedetect @ 0x...] silence_end: 13.187521 | silence_duration: 2.453604
    silence_ends = []
    for match in re.finditer(r'silence_end:\s*([\d.]+)', stderr):
        silence_ends.append(float(match.group(1)))

    # Find first silence_end after min_intro_duration
    # This indicates: intro ended, pause ended, speech starting
    for end_time in silence_ends:
        if end_time >= min_intro_duration:
            return end_time

    return 0.0  # No intro detected


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    position: str = "bottom_bar",
    font_size: int = 24,
    bar_height: int = 80
) -> Path:
    """Burn subtitles into video.

    Args:
        video_path: Input video file
        srt_path: SRT subtitle file
        output_path: Output video file
        position: "overlay" (on video) or "bottom_bar" (black bar below)
        font_size: Subtitle font size
        bar_height: Height of black bar for bottom_bar mode

    Returns:
        Path to output video with burned subtitles
    """
    width, height = get_video_dimensions(video_path)

    # Copy SRT to temp location (ffmpeg subtitles filter needs simple path)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as tmp:
        tmp.write(srt_path.read_text(encoding='utf-8'))
        tmp_srt = Path(tmp.name)

    try:
        if position == "bottom_bar":
            # Add black bar at bottom, put subtitles there
            new_height = height + bar_height
            filter_complex = (
                f"[0:v]pad={width}:{new_height}:0:0:black[padded];"
                f"[padded]subtitles='{tmp_srt}':"
                f"force_style='FontSize={font_size},Alignment=2,"
                f"MarginV={bar_height // 4},PrimaryColour=&HFFFFFF&'"
            )
        else:
            # Overlay on video
            filter_complex = (
                f"subtitles='{tmp_srt}':"
                f"force_style='FontSize={font_size},Alignment=2,"
                f"MarginV=30,PrimaryColour=&HFFFFFF&,"
                f"OutlineColour=&H000000&,Outline=2'"
            )

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", filter_complex,
            "-c:a", "copy",
            str(output_path)
        ]

        subprocess.run(cmd, check=True, capture_output=True)

    finally:
        tmp_srt.unlink(missing_ok=True)

    return output_path


def concatenate_audio_segments(
    segments: list[tuple[Path, float, float]],
    output_path: Path,
    total_duration: float,
    crossfade_ms: int = 50
) -> Path:
    """Concatenate audio segments with proper timing and crossfade.

    Args:
        segments: List of (audio_path, start_time, end_time) tuples
        output_path: Output audio file path
        total_duration: Total duration of the final audio
        crossfade_ms: Crossfade duration between segments in milliseconds

    Returns:
        Path to concatenated audio file
    """
    if not segments:
        # Create silent output
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=mono:d={total_duration}",
            "-acodec", "aac",
            "-b:a", "192k",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create silent base track (stereo for compatibility)
        silent_path = tmpdir / "silent.wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=24000:cl=stereo:d={total_duration}",
            "-acodec", "pcm_s16le",
            str(silent_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        # Build complex filter with crossfade-like transitions
        # Use afade for smooth in/out on each segment
        inputs = ["-i", str(silent_path)]
        filter_parts = []

        fade_ms = crossfade_ms

        for i, (audio_path, start_time, end_time) in enumerate(segments):
            inputs.extend(["-i", str(audio_path)])
            delay_ms = int(start_time * 1000)

            # Apply fade in/out to each segment for smoother transitions
            # Also convert to stereo and resample for consistency
            filter_parts.append(
                f"[{i + 1}]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"afade=t=in:st=0:d={fade_ms/1000},afade=t=out:st={end_time - start_time - fade_ms/1000}:d={fade_ms/1000},"
                f"adelay={delay_ms}|{delay_ms}[s{i}]"
            )

        # Mix all segments together
        if len(segments) == 1:
            mix_input = "[s0]"
        else:
            segment_refs = "".join(f"[s{i}]" for i in range(len(segments)))
            filter_parts.append(f"{segment_refs}amix=inputs={len(segments)}:duration=longest:normalize=0[mixed]")
            mix_input = "[mixed]"

        # Mix with silent base and normalize
        filter_parts.append(f"[0]{mix_input}amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[out]")

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-acodec", "aac",
            "-b:a", "192k",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
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
    original_volume: float = 0.1,
    preserve_non_speech: bool = True
) -> AssemblyResult:
    """Create Hindi dubbed video from segments.

    Args:
        video_path: Original video file
        audio_segments: List of (audio_path, start_time, end_time)
        output_path: Output video path
        keep_original: Keep original audio at low volume during speech
        original_volume: Volume of original audio if kept (0.0-1.0)
        preserve_non_speech: Keep original audio during non-speech (intro/outro/pauses)

    Returns:
        AssemblyResult with output info
    """
    video_duration = get_duration(video_path)

    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        combined_audio_path = Path(tmp.name)

    try:
        if preserve_non_speech:
            # Create mixed audio: original for non-speech, Hindi TTS for speech
            create_mixed_audio(
                video_path,
                audio_segments,
                combined_audio_path,
                video_duration,
                speech_original_volume=original_volume if keep_original else 0.0
            )
            # Already mixed, don't mix again
            result = replace_audio(
                video_path,
                combined_audio_path,
                output_path,
                keep_original_audio=False,
                original_volume=0.0
            )
        else:
            # Old behavior: just concatenate Hindi segments
            concatenate_audio_segments(
                audio_segments,
                combined_audio_path,
                video_duration
            )
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


def create_mixed_audio(
    video_path: Path,
    audio_segments: list[tuple[Path, float, float]],
    output_path: Path,
    total_duration: float,
    speech_original_volume: float = 0.0
) -> Path:
    """Create mixed audio: original for non-speech parts, Hindi TTS for speech.

    Args:
        video_path: Original video to extract audio from
        audio_segments: List of (audio_path, start_time, end_time) for Hindi TTS
        output_path: Output audio file
        total_duration: Total duration of the video
        speech_original_volume: Volume of original audio during speech (0.0 = mute)

    Returns:
        Path to mixed audio file
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract original audio
        original_audio = tmpdir / "original.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "2",
            str(original_audio)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        if not audio_segments:
            # No speech segments, just use original
            subprocess.run(["cp", str(original_audio), str(output_path)], check=True)
            return output_path

        # Build ffmpeg filter to mix audio
        # Strategy: Use original audio as base, duck it during speech, overlay Hindi TTS

        # Sort segments by start time
        sorted_segments = sorted(audio_segments, key=lambda x: x[1])

        # Build volume automation for original audio
        # Full volume during non-speech, reduced during speech
        volume_points = []

        # Start at full volume
        if sorted_segments[0][1] > 0:
            volume_points.append(f"volume=enable='between(t,0,{sorted_segments[0][1]})':volume=1.0")

        for i, (audio_path, start, end) in enumerate(sorted_segments):
            # During speech: use speech_original_volume
            volume_points.append(f"volume=enable='between(t,{start},{end})':volume={speech_original_volume}")

            # After this segment, before next (or end)
            if i < len(sorted_segments) - 1:
                next_start = sorted_segments[i + 1][1]
                if end < next_start:
                    volume_points.append(f"volume=enable='between(t,{end},{next_start})':volume=1.0")
            else:
                if end < total_duration:
                    volume_points.append(f"volume=enable='between(t,{end},{total_duration})':volume=1.0")

        # Build inputs for ffmpeg
        inputs = ["-i", str(original_audio)]
        for audio_path, _, _ in sorted_segments:
            inputs.extend(["-i", str(audio_path)])

        # Build filter complex
        # First, apply volume ducking to original audio
        filter_parts = ["[0]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=stereo"]

        # Add volume points
        for vp in volume_points:
            filter_parts[0] += f",{vp}"
        filter_parts[0] += "[ducked]"

        # Add Hindi TTS segments with delays
        for i, (audio_path, start, end) in enumerate(sorted_segments):
            delay_ms = int(start * 1000)
            filter_parts.append(
                f"[{i + 1}]aresample=24000,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"adelay={delay_ms}|{delay_ms}[tts{i}]"
            )

        # Mix all together
        tts_refs = "".join(f"[tts{i}]" for i in range(len(sorted_segments)))
        filter_parts.append(f"[ducked]{tts_refs}amix=inputs={len(sorted_segments) + 1}:duration=first:normalize=0[mixed]")
        filter_parts.append("[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[out]")

        filter_complex = ";".join(filter_parts)

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-acodec", "aac",
            "-b:a", "192k",
            "-t", str(total_duration),
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Fallback to simpler method if complex filter fails
            print(f"Complex filter failed, using simple method: {e.stderr[:200] if e.stderr else ''}")
            return create_mixed_audio_simple(
                original_audio, audio_segments, output_path, total_duration, speech_original_volume
            )

    return output_path


def create_mixed_audio_simple(
    original_audio: Path,
    audio_segments: list[tuple[Path, float, float]],
    output_path: Path,
    total_duration: float,
    speech_original_volume: float = 0.0
) -> Path:
    """Simpler fallback for mixed audio creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create Hindi TTS track
        hindi_track = tmpdir / "hindi.m4a"
        concatenate_audio_segments(audio_segments, hindi_track, total_duration)

        # Mix with original, ducking original during speech
        # Simple approach: just mix with volume levels
        cmd = [
            "ffmpeg", "-y",
            "-i", str(original_audio),
            "-i", str(hindi_track),
            "-filter_complex",
            f"[0]volume=0.15[orig];[1]volume=1.0[hindi];[orig][hindi]amix=inputs=2:duration=first[out]",
            "-map", "[out]",
            "-acodec", "aac",
            "-b:a", "192k",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    return output_path
