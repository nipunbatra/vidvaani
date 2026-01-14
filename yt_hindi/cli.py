"""CLI for YouTube to Hindi dubbing."""

import argparse
import sys
from pathlib import Path

from .pipeline import run_pipeline
from .tts import generate_samples


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
                           help="Voice (Gemini: Aoede/Charon/Fenrir/Kore/Puck, Edge: male/female)")
    dub_parser.add_argument("-b", "--backend", default="gemini", choices=["gemini", "edge"],
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

    # Samples command
    samples_parser = subparsers.add_parser("samples", help="Generate voice samples")
    samples_parser.add_argument("-o", "--output", default="./samples", help="Output directory")

    args = parser.parse_args()

    if args.command == "dub":
        max_segments = -1 if args.full else args.segments
        result = run_pipeline(
            url=args.url,
            output_dir=Path(args.output),
            voice=args.voice,
            tts_backend=args.backend,
            keep_original_audio=args.keep_original,
            original_volume=args.original_volume,
            whisper_model=args.whisper_model,
            max_segments=max_segments
        )
        print(f"\nOutput video: {result.output_video}")
        print(f"Duration: {result.duration:.1f}s")
        print(f"Segments: {result.segments_count}")

    elif args.command == "samples":
        samples = generate_samples(Path(args.output))
        print(f"\nGenerated {len(samples)} samples:")
        for backend, voice, path in samples:
            print(f"  {backend}/{voice}: {path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
