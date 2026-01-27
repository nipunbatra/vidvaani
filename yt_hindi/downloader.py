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

    # First, extract info without downloading to get video ID
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info["id"]
    title = info["title"]
    duration = info.get("duration", 0)

    # Check if video already exists
    video_path = output_dir / f"{video_id}.mp4"
    already_downloaded = False

    if video_path.exists():
        already_downloaded = True
    else:
        # Try other extensions
        for ext in ['.mkv', '.webm']:
            candidate = output_dir / f"{video_id}{ext}"
            if candidate.exists():
                video_path = candidate
                already_downloaded = True
                break

    # Download only if not already present
    if not already_downloaded:
        def hook(d):
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
            ydl.download([url])

        video_path = output_dir / f"{video_id}.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    else:
        # Signal completion to progress hook
        if progress_hook:
            progress_hook({'status': 'finished', 'filename': str(video_path), '_cached': True})

    audio_path = output_dir / f"{video_id}.wav"

    # Extract audio as WAV for whisper (16kHz mono) - only if not exists
    if not audio_path.exists():
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
