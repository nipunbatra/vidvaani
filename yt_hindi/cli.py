"""CLI for YouTube to Hindi dubbing."""

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline
from .tts import generate_samples
from .downloader import get_video_info
from .video import burn_subtitles


def main():
    parser = argparse.ArgumentParser(
        description="Dub YouTube videos to Hindi"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Dub command
    dub_parser = subparsers.add_parser("dub", help="Dub a YouTube video to Hindi")
    dub_parser.add_argument("url", help="YouTube video URL")
    dub_parser.add_argument("-o", "--output", default="./output", help="Output directory")
    dub_parser.add_argument("-v", "--voice", default="Kore",
                           help="Voice (Gemini: Aoede/Charon/Fenrir/Kore/Puck, Edge: male/female, Sarvam: abhilash/anushka/karun/vidya)")
    dub_parser.add_argument("-b", "--backend", default="gemini", choices=["gemini", "edge", "sarvam"],
                           help="TTS backend")
    dub_parser.add_argument("--keep-original", action="store_true",
                           help="Keep original audio at low volume")
    dub_parser.add_argument("--original-volume", type=float, default=0.1,
                           help="Original audio volume (0.0-1.0)")
    dub_parser.add_argument("-m", "--whisper-model", default="mlx-community/distil-whisper-large-v3",
                           help="Whisper model (tiny/distil-large-v3/large-v3-turbo)")
    dub_parser.add_argument("-n", "--segments", type=int, default=5,
                           help="Number of segments to process (default: 5 for demo, use -1 for all)")
    dub_parser.add_argument("--full", action="store_true",
                           help="Process all segments (same as -n -1)")
    dub_parser.add_argument("--title-folder", action="store_true",
                           help="Create output folder based on video title")
    dub_parser.add_argument("--no-preserve-music", action="store_true",
                           help="Don't preserve original audio during non-speech (intro/outro)")
    dub_parser.add_argument("--intro-offset", type=float, default=None,
                           help="Seconds to skip at start (default: auto-detect intro music, use 0 to disable)")

    # Samples command
    samples_parser = subparsers.add_parser("samples", help="Generate voice samples")
    samples_parser.add_argument("-o", "--output", default="./samples", help="Output directory")

    # Subtitle burn command
    sub_parser = subparsers.add_parser("burn-subs", help="Burn subtitles into video")
    sub_parser.add_argument("video", help="Video file path")
    sub_parser.add_argument("srt", help="SRT subtitle file path")
    sub_parser.add_argument("-o", "--output", help="Output video path (default: video_with_subs.mp4)")
    sub_parser.add_argument("--position", choices=["overlay", "bottom_bar"], default="bottom_bar",
                           help="Subtitle position: overlay on video or in black bar below")
    sub_parser.add_argument("--font-size", type=int, default=24, help="Subtitle font size")
    sub_parser.add_argument("--bar-height", type=int, default=80, help="Black bar height (for bottom_bar)")

    args = parser.parse_args()

    if args.command == "dub":
        max_segments = -1 if args.full else args.segments
        output_dir = Path(args.output)

        # Use video title as folder name if requested
        if args.title_folder:
            info = get_video_info(args.url)
            output_dir = output_dir / info['safe_title']
            print(f"Output folder: {output_dir}")

        result = run_pipeline(
            url=args.url,
            output_dir=output_dir,
            voice=args.voice,
            tts_backend=args.backend,
            keep_original_audio=args.keep_original,
            original_volume=args.original_volume,
            whisper_model=args.whisper_model,
            max_segments=max_segments,
            preserve_non_speech=not args.no_preserve_music,
            intro_offset=args.intro_offset
        )
        print(f"\nOutput video: {result.output_video}")
        print(f"Duration: {result.duration:.1f}s")
        print(f"Segments: {result.segments_count}")

    elif args.command == "samples":
        samples = generate_samples(Path(args.output))
        print(f"\nGenerated {len(samples)} samples:")
        for backend, voice, path in samples:
            print(f"  {backend}/{voice}: {path}")

    elif args.command == "burn-subs":
        video_path = Path(args.video)
        srt_path = Path(args.srt)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = video_path.with_stem(video_path.stem + "_with_subs")

        print(f"Burning subtitles into video...")
        print(f"  Video: {video_path}")
        print(f"  Subtitles: {srt_path}")
        print(f"  Position: {args.position}")

        result = burn_subtitles(
            video_path,
            srt_path,
            output_path,
            position=args.position,
            font_size=args.font_size,
            bar_height=args.bar_height
        )
        print(f"\nOutput: {result}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
