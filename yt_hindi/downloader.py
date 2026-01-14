"""YouTube video downloader using yt-dlp Python API."""

import subprocess
from pathlib import Path
from dataclasses import dataclass

import yt_dlp


@dataclass
class DownloadResult:
    video_path: Path
    audio_path: Path
    title: str
    duration: float


def download_video(url: str, output_dir: Path, progress_hook=None) -> DownloadResult:
    """Download video and extract audio from YouTube URL.

    Args:
        url: YouTube video URL
        output_dir: Directory to save files
        progress_hook: Optional callback for download progress

    Returns:
        DownloadResult with paths to video and audio files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean URL (remove shell escapes)
    url = url.replace("\\", "")

    video_path = None
    info = None

    def hook(d):
        nonlocal video_path
        if d['status'] == 'finished':
            video_path = Path(d['filename'])
        if progress_hook:
            progress_hook(d)

    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'merge_output_format': 'mp4',
        'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
        'progress_hooks': [hook],
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    title = info["title"]
    duration = info.get("duration", 0)

    # The merged file is always {id}.mp4 due to merge_output_format
    video_path = output_dir / f"{video_id}.mp4"

    if not video_path.exists():
        # Try to find any video file with this ID
        for ext in ['.mp4', '.mkv', '.webm']:
            candidate = output_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_path = output_dir / f"{video_id}.wav"

    # Extract audio as WAV for whisper (16kHz mono)
    audio_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]
    subprocess.run(audio_cmd, check=True, capture_output=True)

    return DownloadResult(
        video_path=video_path,
        audio_path=audio_path,
        title=title,
        duration=duration
    )
