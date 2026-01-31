"""Main pipeline for YouTube to Hindi dubbing."""

import json
import time
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
from .video import create_hindi_video, AssemblyResult, detect_intro_offset
from .costs import CostTracker


def generate_srt(segments: list, output_path: Path, language: str = "hi") -> Path:
    """Generate SRT subtitle file from translated segments.

    Args:
        segments: List of TranslatedSegment or dicts with start, end, text/translated
        output_path: Output .srt file path
        language: Language code for the subtitles

    Returns:
        Path to generated SRT file
    """
    def format_time(seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        # Handle both TranslatedSegment objects and dicts
        if hasattr(seg, 'start'):
            start, end = seg.start, seg.end
            text = seg.translated if hasattr(seg, 'translated') else seg.text
        else:
            start, end = seg['start'], seg['end']
            text = seg.get('translated', seg.get('text', ''))

        lines.append(str(i))
        lines.append(f"{format_time(start)} --> {format_time(end)}")
        lines.append(text)
        lines.append("")  # Blank line between entries

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


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
    subtitle_path: Path | None = None
    costs: dict | None = None
    timings: dict | None = None


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
    reuse_translation: bool = True,
    preserve_non_speech: bool = True,
    intro_offset: float | None = None,
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
        reuse_translation: Reuse existing translation if available
        preserve_non_speech: Keep original audio during non-speech (intro/outro/pauses)
        intro_offset: Seconds to skip at start (None = auto-detect intro music)

    Returns:
        PipelineResult with paths to output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset cost tracker
    CostTracker.reset()

    # Timing tracker
    timings = {}
    pipeline_start = time.time()

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
        step_start = time.time()
        download_task = progress.add_task("[cyan]Downloading video...", total=100)

        def download_progress(d):
            if d['status'] == 'downloading':
                total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
                downloaded = d.get('downloaded_bytes', 0)
                if total > 0:
                    progress.update(download_task, completed=int(downloaded / total * 100))

        download_result = download_video(url, output_dir, progress_hook=download_progress)
        progress.update(download_task, completed=100, description=f"[green]Downloaded: {download_result.title[:40]}...")
        timings['download'] = time.time() - step_start

        # Auto-detect intro offset if not specified
        step_start = time.time()
        if intro_offset is None:
            intro_offset = detect_intro_offset(download_result.video_path)
            if intro_offset > 0:
                console.print(f"[yellow]Auto-detected intro: {intro_offset:.1f}s[/yellow]")
        timings['intro_detect'] = time.time() - step_start

        # Check for existing translation to reuse
        translation_path = output_dir / f"{download_result.video_path.stem}_transcript_hi.json"
        transcript_path = output_dir / f"{download_result.video_path.stem}_transcript_en.json"
        translated = None

        if reuse_translation and translation_path.exists():
            # Load existing translation
            with open(translation_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            translated = [
                TranslatedSegment(
                    start=s["start"],
                    end=s["end"],
                    original=s["original"],
                    translated=s["text"]
                )
                for s in data["segments"]
            ]
            console.print(f"[yellow]Reusing existing translation: {len(translated)} segments[/yellow]")
            timings['transcribe'] = 0.0
            timings['translate'] = 0.0
        else:
            # Step 2: Transcribe audio with real progress
            step_start = time.time()
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
            timings['transcribe'] = time.time() - step_start

            # Save English transcript
            if save_intermediate:
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
            step_start = time.time()
            translate_task = progress.add_task("[cyan]Translating to Hindi...", total=None)
            translated = translate_segments(
                grouped_segments,
                source_lang="English",
                target_lang="Hindi",
                model=translation_model
            )
            progress.update(translate_task, completed=100, total=100,
                           description=f"[green]Translated: {len(translated)} segments")
            timings['translate'] = time.time() - step_start

            # Save Hindi transcript
            if save_intermediate:
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

        # Generate SRT subtitle file (includes all segments)
        srt_path = output_dir / f"{download_result.video_path.stem}_hindi.srt"
        generate_srt(translated, srt_path, language="hi")
        console.print(f"[yellow]Generated subtitles: {srt_path.name}[/yellow]")

        # Adjust segments for intro offset (keeps original audio during intro)
        if intro_offset > 0:
            adjusted_translated = []
            for s in translated:
                if s.end <= intro_offset:
                    # Entire segment is in intro - skip it
                    continue
                elif s.start < intro_offset:
                    # Segment spans intro boundary - adjust start time
                    # Create new segment starting at intro_offset
                    adjusted_translated.append(TranslatedSegment(
                        start=intro_offset,
                        end=s.end,
                        original=s.original,
                        translated=s.translated
                    ))
                else:
                    # Segment is after intro - keep as is
                    adjusted_translated.append(s)

            skipped = len(translated) - len(adjusted_translated)
            adjusted = len([s for s in translated if s.start < intro_offset and s.end > intro_offset])
            if skipped > 0 or adjusted > 0:
                console.print(f"[yellow]Intro ({intro_offset}s): skipped {skipped}, adjusted {adjusted} segments[/yellow]")
            translated = adjusted_translated

        # Step 4: Generate Hindi TTS in parallel
        step_start = time.time()
        tts_task = progress.add_task(
            f"[cyan]Generating Hindi speech ({voice}, {tts_workers} workers)...",
            total=len(translated)
        )

        # Use voice-specific TTS directory
        tts_dir = output_dir / f"tts_segments_{voice.lower()}"
        tts_dir.mkdir(exist_ok=True)

        # Build TTS jobs (skip already completed segments and very short text)
        jobs = []
        cached_count = 0
        skipped_short = 0
        for i, seg in enumerate(translated):
            output_path = tts_dir / f"segment_{i:04d}.mp3"
            # Skip very short segments (just punctuation or < 5 chars of actual text)
            clean_text = ''.join(c for c in seg.translated if c.isalnum() or c.isspace())
            if len(clean_text.strip()) < 5:
                skipped_short += 1
                continue
            if output_path.exists():
                cached_count += 1
            else:
                jobs.append(TTSJob(
                    index=len(jobs),  # Use sequential index for the jobs list
                    text=seg.translated,
                    start=seg.start,
                    end=seg.end,
                    output_path=output_path
                ))

        if skipped_short > 0:
            console.print(f"[yellow]Skipped {skipped_short} segments with very short text[/yellow]")

        if cached_count > 0:
            console.print(f"[yellow]Skipping {cached_count} already generated TTS segments[/yellow]")

        def tts_progress(completed: int, total: int):
            progress.update(tts_task, completed=completed,
                          description=f"[cyan]TTS: {completed}/{total} segments (parallel)")

        # Run TTS in parallel for remaining jobs
        if jobs:
            synthesize_batch_parallel(
                jobs,
                voice=voice,
                backend=tts_backend,
                max_workers=min(tts_workers, 3),  # Reduce workers to avoid rate limits
                progress_callback=tts_progress
            )

        # Collect all audio segments (including cached ones)
        audio_segments = [
            (tts_dir / f"segment_{i:04d}.mp3", seg.start, seg.end)
            for i, seg in enumerate(translated)
            if (tts_dir / f"segment_{i:04d}.mp3").exists()
        ]

        progress.update(tts_task, completed=len(translated),
                       description=f"[green]Generated: {len(audio_segments)} audio segments")
        timings['tts'] = time.time() - step_start

        # Step 5: Assemble final video
        step_start = time.time()
        assemble_task = progress.add_task("[cyan]Assembling Hindi video...", total=None)
        output_video = output_dir / f"{download_result.video_path.stem}_hindi_{voice.lower()}.mp4"

        assembly_result = create_hindi_video(
            download_result.video_path,
            audio_segments,
            output_video,
            keep_original=keep_original_audio,
            original_volume=original_volume,
            preserve_non_speech=preserve_non_speech
        )
        progress.update(assemble_task, completed=100, total=100,
                       description=f"[green]Created: {output_video.name}")
        timings['assemble'] = time.time() - step_start

    # Cleanup intermediate audio file
    download_result.audio_path.unlink(missing_ok=True)

    # Calculate total time
    timings['total'] = time.time() - pipeline_start

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

    # Print timing breakdown
    timing_table = Table(title="Timing Breakdown", show_header=True, border_style="cyan")
    timing_table.add_column("Step", style="cyan")
    timing_table.add_column("Time", style="white", justify="right")
    timing_table.add_column("% of Total", style="yellow", justify="right")

    total_time = timings['total']
    for step, label in [
        ('download', 'Download'),
        ('intro_detect', 'Intro Detection'),
        ('transcribe', 'Transcription'),
        ('translate', 'Translation'),
        ('tts', 'TTS Generation'),
        ('assemble', 'Video Assembly'),
    ]:
        if step in timings:
            t = timings[step]
            pct = (t / total_time * 100) if total_time > 0 else 0
            timing_table.add_row(label, f"{t:.1f}s", f"{pct:.1f}%")

    timing_table.add_row("[bold]Total[/bold]", f"[bold]{total_time:.1f}s[/bold]", "[bold]100%[/bold]")
    console.print(timing_table)

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
        subtitle_path=srt_path,
        costs=costs,
        timings=timings
    )
