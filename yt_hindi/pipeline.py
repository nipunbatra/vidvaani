"""Main pipeline for YouTube to Hindi dubbing."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from .downloader import download_video, DownloadResult
from .transcriber import transcribe, Transcript, group_segments_by_duration
from .translator import translate_segments, TranslatedSegment
from .tts import synthesize_batch_parallel, TTSJob
from .video import create_hindi_video, AssemblyResult
from .costs import CostTracker


console = Console()


@dataclass
class PipelineResult:
    input_url: str
    output_video: Path
    title: str
    duration: float
    segments_count: int
    transcript_path: Path | None = None
    translation_path: Path | None = None
    costs: dict | None = None


def run_pipeline(
    url: str,
    output_dir: Path,
    voice: str = "Kore",
    tts_backend: Literal["gemini", "edge"] = "gemini",
    keep_original_audio: bool = False,
    original_volume: float = 0.1,
    save_intermediate: bool = True,
    whisper_model: str = "mlx-community/distil-whisper-large-v3",
    translation_model: str = "gemini-2.0-flash",
    max_segment_duration: float = 15.0,
    tts_workers: int = 5,
    max_segments: int = 5,
) -> PipelineResult:
    """Run the complete YouTube to Hindi dubbing pipeline.

    Args:
        url: YouTube video URL
        output_dir: Directory for output files
        voice: TTS voice (Gemini: Aoede/Charon/Fenrir/Kore/Puck, Edge: male/female)
        tts_backend: TTS backend to use
        keep_original_audio: Keep original audio at low volume
        original_volume: Volume of original audio if kept
        save_intermediate: Save transcript and translation files
        whisper_model: Whisper model for transcription
        translation_model: Gemini model for translation
        max_segment_duration: Max duration per segment
        tts_workers: Number of parallel TTS workers
        max_segments: Max segments to process (-1 for all, default 5 for demo)

    Returns:
        PipelineResult with paths to output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset cost tracker
    CostTracker.reset()

    console.print(Panel.fit(f"[bold blue]YouTube → Hindi Dubbing[/bold blue]\n{url}", border_style="blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:

        # Step 1: Download video
        download_task = progress.add_task("[cyan]Downloading video...", total=100)

        def download_progress(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                if total > 0:
                    progress.update(download_task, completed=int(downloaded / total * 100))

        download_result = download_video(url, output_dir, progress_hook=download_progress)
        progress.update(download_task, completed=100, description=f"[green]Downloaded: {download_result.title[:40]}...")

        # Step 2: Transcribe audio with real progress
        audio_duration = download_result.duration
        transcribe_task = progress.add_task(
            f"[cyan]Transcribing {audio_duration:.0f}s audio...",
            total=int(audio_duration) if audio_duration > 0 else 100
        )

        def transcribe_progress(current: int, total: int):
            progress.update(transcribe_task, completed=current, total=total,
                          description=f"[cyan]Transcribing: {current}s / {total}s")

        transcript = transcribe(download_result.audio_path, model=whisper_model, progress_callback=transcribe_progress)
        grouped_segments = group_segments_by_duration(transcript.segments, max_segment_duration)

        # Limit segments for demo mode
        total_segments = len(grouped_segments)
        if max_segments > 0 and len(grouped_segments) > max_segments:
            grouped_segments = grouped_segments[:max_segments]
            progress.update(transcribe_task, completed=100, total=100,
                           description=f"[green]Transcribed: {len(grouped_segments)}/{total_segments} segments (demo mode)")
        else:
            progress.update(transcribe_task, completed=100, total=100,
                           description=f"[green]Transcribed: {len(grouped_segments)} segments")

        # Save English transcript
        transcript_path = None
        if save_intermediate:
            transcript_path = output_dir / f"{download_result.video_path.stem}_transcript_en.json"
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump({
                    "language": transcript.language,
                    "text": transcript.text,
                    "segments": [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in grouped_segments
                    ]
                }, f, ensure_ascii=False, indent=2)

        # Step 3: Translate to Hindi
        translate_task = progress.add_task("[cyan]Translating to Hindi...", total=None)
        translated = translate_segments(
            grouped_segments,
            source_lang="English",
            target_lang="Hindi",
            model=translation_model
        )
        progress.update(translate_task, completed=100, total=100,
                       description=f"[green]Translated: {len(translated)} segments")

        # Save Hindi transcript
        translation_path = None
        if save_intermediate:
            translation_path = output_dir / f"{download_result.video_path.stem}_transcript_hi.json"
            with open(translation_path, "w", encoding="utf-8") as f:
                json.dump({
                    "language": "hi",
                    "segments": [
                        {
                            "start": s.start,
                            "end": s.end,
                            "original": s.original,
                            "text": s.translated
                        }
                        for s in translated
                    ]
                }, f, ensure_ascii=False, indent=2)

        # Step 4: Generate Hindi TTS in parallel
        tts_task = progress.add_task(
            f"[cyan]Generating Hindi speech ({voice}, {tts_workers} workers)...",
            total=len(translated)
        )

        tts_dir = output_dir / "tts_segments"
        tts_dir.mkdir(exist_ok=True)

        # Build TTS jobs
        jobs = [
            TTSJob(
                index=i,
                text=seg.translated,
                start=seg.start,
                end=seg.end,
                output_path=tts_dir / f"segment_{i:04d}.mp3"
            )
            for i, seg in enumerate(translated)
        ]

        def tts_progress(completed: int, total: int):
            progress.update(tts_task, completed=completed,
                          description=f"[cyan]TTS: {completed}/{total} segments (parallel)")

        # Run TTS in parallel
        audio_segments = synthesize_batch_parallel(
            jobs,
            voice=voice,
            backend=tts_backend,
            max_workers=tts_workers,
            progress_callback=tts_progress
        )

        progress.update(tts_task, completed=len(translated),
                       description=f"[green]Generated: {len(audio_segments)} audio segments")

        # Step 5: Assemble final video
        assemble_task = progress.add_task("[cyan]Assembling Hindi video...", total=None)
        output_video = output_dir / f"{download_result.video_path.stem}_hindi.mp4"

        assembly_result = create_hindi_video(
            download_result.video_path,
            audio_segments,
            output_video,
            keep_original=keep_original_audio,
            original_volume=original_volume
        )
        progress.update(assemble_task, completed=100, total=100,
                       description=f"[green]Created: {output_video.name}")

    # Cleanup intermediate audio file
    download_result.audio_path.unlink(missing_ok=True)

    # Get costs
    costs = CostTracker.get().summary()

    # Print summary
    console.print()
    table = Table(title="Pipeline Complete", show_header=False, border_style="green")
    table.add_row("Output", str(output_video))
    table.add_row("Duration", f"{assembly_result.duration:.1f}s")
    table.add_row("Segments", str(len(audio_segments)))
    if transcript_path:
        table.add_row("English transcript", str(transcript_path))
    if translation_path:
        table.add_row("Hindi transcript", str(translation_path))
    console.print(table)

    # Print costs
    cost_table = Table(title="Gemini API Costs", show_header=True, border_style="yellow")
    cost_table.add_column("Service", style="cyan")
    cost_table.add_column("Usage", style="white")
    cost_table.add_column("Cost", style="green")

    cost_table.add_row(
        "Translation",
        f"{costs['translation']['input_tokens']:,} in / {costs['translation']['output_tokens']:,} out tokens",
        f"${costs['translation']['cost_usd']:.6f}"
    )
    if 'tts_gemini' in costs:
        cost_table.add_row(
            "TTS (Gemini)",
            f"{costs['tts_gemini']['characters']:,} chars ({costs['tts_gemini']['calls']} calls)",
            f"${costs['tts_gemini']['cost_usd']:.6f}"
        )
    if 'tts_sarvam' in costs:
        cost_table.add_row(
            "TTS (Sarvam)",
            f"{costs['tts_sarvam']['characters']:,} chars ({costs['tts_sarvam']['calls']} calls)",
            f"Rs {costs['tts_sarvam']['cost_inr']:.2f} (${costs['tts_sarvam']['cost_usd']:.6f})"
        )
    cost_table.add_row(
        "[bold]Total[/bold]",
        "",
        f"[bold]${costs['total_cost_usd']:.6f}[/bold]"
    )
    console.print(cost_table)

    return PipelineResult(
        input_url=url,
        output_video=output_video,
        title=download_result.title,
        duration=assembly_result.duration,
        segments_count=len(audio_segments),
        transcript_path=transcript_path,
        translation_path=translation_path,
        costs=costs
    )
